import os
import sys
import torch
import hydra
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import logging 
from sklearn.metrics import classification_report, confusion_matrix

torch.set_printoptions(precision=10)

from datasets.radar_scenes import RadarScenesDataset
from models.ae import ProbabilisticAE
from models.hmm import SceneTransitionModel
from models.gnn import GNNPointSegmenter
from torch_geometric.data import Data as PyGData, Batch
from torch_geometric.nn import knn_graph
from torch_geometric.nn import global_mean_pool
from utils.metrics import classification_metrics

def collate_fn_pyg_batch(batch):
    """Custom collate function for PyG.Batch creation."""
    data_list = []
    for points, uncertainties, point_labels, timestamp in batch:
        data_list.append(PyGData(x=points, y=point_labels, uncertainties=uncertainties, timestamp=timestamp))
    return Batch.from_data_list(data_list)

class FocalLoss(nn.Module):
    """Focal Loss implementation for imbalanced classification."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', epsilon=1e-8):
        super(FocalLoss, self).__init__()
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction
        self.epsilon = epsilon
        self.clamp_min_prob = 1e-6
        self.clamp_max_prob = 1. - 1e-6

    def forward(self, inputs, targets):
        if self.alpha is not None:
            alpha_on_device = self.alpha.to(inputs.device)
        else:
            alpha_on_device = None

        log_probs = F.log_softmax(inputs, dim=1)
        log_pt = torch.gather(log_probs, 1, targets.view(-1, 1)).squeeze(1)
        pt = torch.exp(log_pt)
        
        pt = pt.clamp(self.clamp_min_prob, self.clamp_max_prob)
        focal_term = (1 - pt)**self.gamma
        ce_loss = -torch.log(pt)

        if alpha_on_device is not None:
            alpha_t = torch.gather(alpha_on_device, 0, targets.view(-1)).squeeze(0)
            loss = alpha_t * focal_term * ce_loss
        else:
            loss = focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    log_dir_base = os.path.join(os.getcwd(), 'tensorboard_scene')
    writer = SummaryWriter(log_dir_base)
    print(f"TensorBoard (scene): {log_dir_base}")

    from hydra.utils import get_original_cwd
    root = get_original_cwd()

    log_output_dir = os.path.join(root, cfg.sim.output_dir)
    os.makedirs(log_output_dir, exist_ok=True)
    debug_log_path = os.path.join(log_output_dir, cfg.scene.debug_log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(debug_log_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Detailed batch diagnostics logged to: {debug_log_path}")

    hmm_full_path = os.path.join(root, cfg.scene.hmm_model_path)
    gnn_full_path = os.path.join(root, cfg.scene.gnn_model_path)

    if os.path.exists(hmm_full_path):
        os.remove(hmm_full_path)
        logger.info(f"Deleted old HMM checkpoint: {hmm_full_path}")
    if os.path.exists(gnn_full_path):
        os.remove(gnn_full_path)
        logger.info(f"Deleted old GNNPointSegmenter checkpoint: {gnn_full_path}")

    ds = RadarScenesDataset(
        data_dir=cfg.data.root,
        calibration_params=cfg.calibration_params,
        num_classes=cfg.scene.num_classes
    )
    split = int(len(ds) * cfg.data.train_frac)
    train_ds, val_ds = torch.utils.data.random_split(ds, [split, len(ds)-split])
    logger.info(f"Dataset split: Train samples = {len(train_ds)}, Validation samples = {len(val_ds)}")

    ae = ProbabilisticAE(
        input_dim=cfg.sensors.input_dim,
        latent_dim=cfg.sensors.latent_dim,
        hidden_dims=cfg.sensors.hidden_dims,
        num_classes=cfg.scene.num_classes
    ).to(device)
    
    ae_full_path = os.path.join(root, cfg.sensors.ae_model_path)

    if os.path.exists(ae_full_path):
        try:
            ae.load_state_dict(torch.load(ae_full_path, map_location=device))
            ae.eval()
            logger.info(f"Loaded AE checkpoint from: {ae_full_path}")
        except RuntimeError as e:
            logger.error(f"CRITICAL ERROR: AE model architecture mismatch during loading. Error: {e}")
            sys.exit(1)
    else:
        raise FileNotFoundError(
            f"AE model not found at '{ae_full_path}'. Verify model existence."
        )

    point_segmenter_gnn = GNNPointSegmenter(
        point_feature_dim=cfg.sensors.latent_dim,
        scene_context_dim=cfg.scene.num_states,
        hidden_dim=cfg.scene.hidden_dim,
        num_point_classes=cfg.scene.num_classes,
        gnn_hidden_layers=cfg.scene.gnn_hidden_layers
    ).to(device)
    
    hmm = SceneTransitionModel(cfg.scene.num_states, cfg.sensors.latent_dim).to(device)

    optimizer_hmm = torch.optim.Adam(hmm.parameters(), lr=cfg.scene.lr)
    optimizer_gnn = torch.optim.Adam(point_segmenter_gnn.parameters(), lr=cfg.scene.lr, weight_decay=cfg.scene.gnn_weight_decay)

    scheduler_gnn = lr_scheduler.ReduceLROnPlateau(
        optimizer_gnn, 
        mode='min', 
        factor=cfg.scene.scheduler_factor,
        patience=cfg.scene.scheduler_patience,
    )

    logger.info("\n--- Calculating class weights for loss and sampler ---")
    all_train_point_labels = []
    temp_train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=cfg.scene.num_workers)
    
    for i, (points, uncertainties, point_labels, timestamp) in enumerate(tqdm(temp_train_loader, desc="Collecting training labels")):
        if point_labels.numel() > 0:
            all_train_point_labels.append(point_labels.squeeze(0).cpu().numpy())

    class_base_sample_weights = torch.zeros(cfg.scene.num_classes, dtype=torch.float32)

    if not all_train_point_labels:
        logger.warning("No point labels collected. Class weights will be uniform.")
        class_weights_for_loss = torch.ones(cfg.scene.num_classes).to(device) 
        sample_weights = torch.ones(len(train_ds))
    else:
        stacked_train_labels = np.concatenate(all_train_point_labels)
        class_counts = np.bincount(stacked_train_labels, minlength=cfg.scene.num_classes)
        
        epsilon = 1e-8 
        class_counts_tensor = torch.tensor(class_counts, dtype=torch.float32)
        
        total_points_in_dataset = class_counts_tensor.sum()
        num_classes_in_cfg = cfg.scene.num_classes
        
        class_weights_for_loss = torch.zeros(num_classes_in_cfg, dtype=torch.float32)
        for c in range(num_classes_in_cfg):
            if class_counts[c] > 0:
                class_weights_for_loss[c] = total_points_in_dataset / (num_classes_in_cfg * class_counts_tensor[c])
            else:
                class_weights_for_loss[c] = epsilon
        
        if cfg.scene.num_classes > 11: 
            class_weights_for_loss[11] = getattr(cfg.scene, 'weight_for_static_override', 1.0) 

        if hasattr(cfg.scene, 'max_class_weight'):
            class_weights_for_loss = torch.clamp(class_weights_for_loss, max=cfg.scene.max_class_weight)
        
        class_weights_for_loss = class_weights_for_loss.to(device)
        
        logger.info(f"Class counts: {class_counts}")
        logger.info(f"Calculated class weights (for loss): {class_weights_for_loss.cpu().numpy()}")
        
        target_samples_per_class = getattr(cfg.scene, 'target_samples_per_class', 10000)
        for c in range(num_classes_in_cfg):
            if class_counts[c] > 0:
                class_base_sample_weights[c] = target_samples_per_class / class_counts[c]
            else:
                class_base_sample_weights[c] = epsilon
        sample_weights_list = []
        for i in tqdm(range(len(train_ds)), desc="Calculating sample weights"):
            _, _, frame_point_labels, _ = train_ds[i]
            if frame_point_labels.numel() == 0:
                sample_weights_list.append(epsilon)
                continue
            
            unique_labels_in_frame = torch.unique(frame_point_labels).long()
            if unique_labels_in_frame.numel() > 0:
                frame_weight = class_base_sample_weights[unique_labels_in_frame].sum().item()
                sample_weights_list.append(frame_weight)
            else:
                sample_weights_list.append(epsilon)

        sample_weights = torch.tensor(sample_weights_list, dtype=torch.float32)
        sample_weights[sample_weights < epsilon] = epsilon
        logger.info(f"Calculated sample weights (first 10): {sample_weights[:10].cpu().numpy()}")

    if getattr(cfg.scene, 'use_focal_loss', False):
        criterion_gnn = FocalLoss(alpha=class_weights_for_loss, gamma=getattr(cfg.scene, 'focal_loss_gamma', 2.0))
        logger.info(f"Using FocalLoss (alpha={class_weights_for_loss.cpu().numpy()}, gamma={getattr(cfg.scene, 'focal_loss_gamma', 2.0)})")
    else:
        criterion_gnn = nn.CrossEntropyLoss(weight=class_weights_for_loss, reduction='mean')
        logger.info(f"Using CrossEntropyLoss (weight={class_weights_for_loss.cpu().numpy()})")

    sampler_total_multiplier = getattr(cfg.scene, 'sampler_total_multiplier', 1.0)
    num_samples_to_draw = int(target_samples_per_class * cfg.scene.num_classes * sampler_total_multiplier)
    train_sampler = WeightedRandomSampler(
        sample_weights, 
        num_samples=num_samples_to_draw, 
        replacement=True
    )
    logger.info(f"WeightedRandomSampler created with {num_samples_to_draw} samples per epoch (targeting ~{target_samples_per_class} per class).")

    train_loader = DataLoader(train_ds, batch_size=cfg.scene.batch_size, shuffle=False,
                              num_workers=cfg.scene.num_workers, collate_fn=collate_fn_pyg_batch,
                              sampler=train_sampler)
    
    val_loader = DataLoader(val_ds, batch_size=cfg.scene.batch_size, shuffle=False,
                            num_workers=cfg.scene.num_workers, collate_fn=collate_fn_pyg_batch)

    logger.info(f"\n--- Training HMM and GNNPointSegmenter for {cfg.scene.epochs} epochs ---")
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = getattr(cfg.scene, 'early_stopping_patience', 10)

    for epoch in range(cfg.scene.epochs):
        hmm.train()
        point_segmenter_gnn.train()
        
        total_hmm_loss = 0
        total_gnn_loss = 0
        total_points_processed = 0
        
        train_true_point_labels_epoch = []
        train_pred_point_labels_epoch = []

        for batch_idx, data_batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            data_batch = data_batch.to(device)

            if data_batch.num_nodes == 0:
                logger.warning(f"Batch {batch_idx+1} is empty. Skipping.")
                continue

            with torch.no_grad():
                _, mu_per_point, _, _, _ = ae(data_batch.x)

            mu_scene_batch = global_mean_pool(mu_per_point, data_batch.batch)
            prior_states = torch.full((mu_scene_batch.size(0), cfg.scene.num_states), 1.0/cfg.scene.num_states, device=device)
            scene_context_for_gnn = hmm(prior_states, mu_scene_batch)
            
            num_neighbors = getattr(cfg.gnn, 'k_neighbors', 8)
            edge_index = knn_graph(mu_per_point, k=num_neighbors, batch=data_batch.batch, loop=False)

            pred_logits_gnn = point_segmenter_gnn(
                x=mu_per_point,
                edge_index=edge_index,
                scene_context=scene_context_for_gnn,
                batch=data_batch.batch
            )

            if torch.isnan(data_batch.y).any() or torch.isinf(data_batch.y).any():
                logger.error(f"CRITICAL ERROR: Target labels (data_batch.y) contain NaN/Inf in Batch {batch_idx+1}. Halting training.")
                sys.exit(1)

            if data_batch.y.max() >= cfg.scene.num_classes or data_batch.y.min() < 0:
                logger.error(f"CRITICAL ERROR: Target labels (data_batch.y) out of range [0, {cfg.scene.num_classes-1}] in Batch {batch_idx+1}. Max: {data_batch.y.max()}, Min: {data_batch.y.min()}. Halting training.")
                sys.exit(1)

            gnn_loss = criterion_gnn(pred_logits_gnn, data_batch.y)
            
            optimizer_gnn.zero_grad()
            gnn_loss.backward()
            torch.nn.utils.clip_grad_norm_(point_segmenter_gnn.parameters(), max_norm=1.0)
            optimizer_gnn.step()
            
            total_gnn_loss += gnn_loss.item() * data_batch.num_nodes
            total_points_processed += data_batch.num_nodes

            pred_class_ids_batch = torch.argmax(pred_logits_gnn, dim=-1)

            batch_pred_counts = np.bincount(pred_class_ids_batch.cpu().numpy(),
                                            minlength=cfg.scene.num_classes)
            logger.info(f"Batch {batch_idx+1}: Predictions per class: {batch_pred_counts}")
            
            unique_true_labels = torch.unique(data_batch.y).cpu().numpy()
            unique_pred_labels = torch.unique(pred_class_ids_batch).cpu().numpy()
            
            if torch.isnan(pred_logits_gnn).any() or torch.isinf(pred_logits_gnn).any():
                logger.error(f"CRITICAL ERROR: Logits are NaN/Inf in Batch {batch_idx+1}. Halting training.")
                sys.exit(1)

            probs_batch = F.softmax(pred_logits_gnn, dim=1)
            pt_batch = torch.gather(probs_batch, 1, data_batch.y.view(-1, 1)).squeeze(1)
            
            writer.add_scalar("Train/pt_mean", pt_batch.mean().item(), epoch * len(train_loader) + batch_idx)

            if cfg.scene.num_classes > 11 and (data_batch.y == 11).any():
                pt_class11_batch = pt_batch[data_batch.y == 11]
                if pt_class11_batch.numel() > 0:
                    writer.add_histogram("Train/pt_class_11", pt_class11_batch, epoch * len(train_loader) + batch_idx)
                    logger.info(f"Batch {batch_idx+1}: pt (Class 11) Min: {pt_class11_batch.min().item():.10f}, Max: {pt_class11_batch.max().item():.10f}, Mean: {pt_class11_batch.mean().item():.10f}")

            if torch.isnan(probs_batch).any() or torch.isinf(probs_batch).any():
                logger.error(f"CRITICAL ERROR: Probabilities are NaN/Inf in Batch {batch_idx+1}. Halting training.")
                sys.exit(1)

            if torch.isnan(pt_batch).any() or torch.isinf(pt_batch).any():
                logger.error(f"CRITICAL ERROR: pt is NaN/Inf in Batch {batch_idx+1}. Halting training.")
                sys.exit(1)

            if torch.isnan(gnn_loss) or torch.isinf(gnn_loss):
                logger.error(f"CRITICAL ERROR: GNN Loss is NaN/Inf in Batch {batch_idx+1}. Halting training.")
                sys.exit(1)

            if (batch_idx + 1) % 10 == 0 or batch_idx == len(train_loader) - 1:
                logger.info(f"Epoch {epoch+1} Batch {batch_idx+1}: True labels: {unique_true_labels}, Predicted labels: {unique_pred_labels}")
                logger.info(f"Batch {batch_idx+1}: Logits Min: {pred_logits_gnn.min().item():.10f}, Max: {pred_logits_gnn.max().item():.10f}, Mean: {pred_logits_gnn.mean().item():.10f}")
                logger.info(f"Batch {batch_idx+1}: Probabilities Min: {probs_batch.min().item():.10f}, Max: {probs_batch.max().item():.10f}, Mean: {probs_batch.mean().item():.10f}")
                logger.info(f"Batch {batch_idx+1}: pt Min: {pt_batch.min().item():.10f}, Max: {pt_batch.max().item():.10f}, Mean: {pt_batch.mean().item():.10f}")
                logger.info(f"Batch {batch_idx+1}: GNN Loss: {gnn_loss.item():.10f}")

            train_true_point_labels_epoch.append(data_batch.y.cpu().numpy())
            train_pred_point_labels_epoch.append(pred_class_ids_batch.cpu().numpy())

        if total_points_processed > 0:
            avg_gnn_loss = total_gnn_loss / total_points_processed
            logger.info(f"Epoch {epoch+1}/{cfg.scene.epochs}, Avg GNN Loss: {avg_gnn_loss:.10f}")
            writer.add_scalar("Train/GNN_Point_Segmenter_Loss", avg_gnn_loss, epoch)

            if len(train_true_point_labels_epoch) > 0:
                stacked_train_true_labels = np.concatenate(train_true_point_labels_epoch)
                stacked_train_pred_labels = np.concatenate(train_pred_point_labels_epoch)
                
                train_overall_metrics = classification_metrics(stacked_train_true_labels, stacked_train_pred_labels, average='macro')
                logger.info(f"Epoch {epoch+1} Train Accuracy: {train_overall_metrics['accuracy']:.3f}, Train Macro F1: {train_overall_metrics['f1']:.3f}")
                writer.add_scalar("Train/Overall_Point_Accuracy", train_overall_metrics['accuracy'], epoch)
                writer.add_scalar("Train/Overall_Point_Macro_F1", train_overall_metrics['f1'], epoch)

                all_preds = np.concatenate(train_pred_point_labels_epoch)
                epoch_pred_counts = np.bincount(all_preds, minlength=cfg.scene.num_classes)
                logger.info(f"Epoch {epoch+1}: Total predictions per class (train): {epoch_pred_counts}")
                writer.add_histogram("Train/Predictions_per_Class", epoch_pred_counts, epoch)
                
                filtered_preds_train = all_preds[all_preds != 11]
                epoch_pred_counts_no_static_train = np.bincount(filtered_preds_train, minlength=cfg.scene.num_classes - 1) 
                logger.info(f"Epoch {epoch+1}: Total predictions per class (EXCLUDING STATIC, train): {epoch_pred_counts_no_static_train}")
                writer.add_histogram("Train/Predictions_per_Class_no_static", epoch_pred_counts_no_static_train, epoch)

                per_class_f1_train = classification_metrics(stacked_train_true_labels, stacked_train_pred_labels, average=None)['f1']
                per_class_precision_train = classification_metrics(stacked_train_true_labels, stacked_train_pred_labels, average=None)['precision']
                per_class_recall_train = classification_metrics(stacked_train_true_labels, stacked_train_pred_labels, average=None)['recall']
                
                for cls in range(cfg.scene.num_classes):
                    if cls < len(per_class_f1_train):
                        writer.add_scalar(f"Train/Point_Class{cls}_F1", per_class_f1_train[cls], epoch)
                        writer.add_scalar(f"Train/Point_Class{cls}_Precision", per_class_precision_train[cls], epoch)
                        writer.add_scalar(f"Train/Point_Class{cls}_Recall", per_class_recall_train[cls], epoch)
            else:
                logger.warning(f"Epoch {epoch+1}/{cfg.scene.epochs}, No points processed in training.")

        hmm_full_path = os.path.join(root, cfg.scene.hmm_model_path)
        gnn_full_path = os.path.join(root, cfg.scene.gnn_model_path)

        torch.save(hmm.state_dict(), hmm_full_path)
        logger.info(f"Saved HMM model to: {hmm_full_path}")
        torch.save(point_segmenter_gnn.state_dict(), gnn_full_path)
        logger.info(f"Saved GNNPointSegmenter model to: {gnn_full_path}")

        logger.info("\n--- Running Validation after HMM and GNN training ---")
        hmm.eval()
        point_segmenter_gnn.eval()

        val_true_point_labels = []
        val_pred_point_labels = []
        val_total_loss = 0
        val_total_points = 0

        with torch.no_grad():
            for batch_idx, data_batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} Validation")):
                data_batch = data_batch.to(device)

                if data_batch.num_nodes == 0: continue

                _, mu_per_point, _, _, _ = ae(data_batch.x)

                mu_scene_batch = global_mean_pool(mu_per_point, data_batch.batch)
                prior_states = torch.full((mu_scene_batch.size(0), cfg.scene.num_states), 1.0/cfg.scene.num_states, device=device)
                scene_context = hmm(prior_states, mu_scene_batch)

                num_neighbors = getattr(cfg.gnn, 'k_neighbors', 8)
                edge_index = knn_graph(mu_per_point, k=num_neighbors, batch=data_batch.batch, loop=False)

                pred_logits = point_segmenter_gnn(
                    x=mu_per_point,
                    edge_index=edge_index,
                    scene_context=scene_context,
                    batch=data_batch.batch
                )

                val_loss = criterion_gnn(pred_logits, data_batch.y) 
                val_total_loss += val_loss.item() * data_batch.num_nodes
                val_total_points += data_batch.num_nodes

                pred_class_ids = torch.argmax(pred_logits, dim=-1).cpu().numpy()
                val_true_point_labels.append(data_batch.y.cpu().numpy())
                val_pred_point_labels.append(pred_class_ids)
        
        if val_total_points > 0:
            stacked_true_labels_val = np.concatenate(val_true_point_labels)
            stacked_pred_labels_val = np.concatenate(val_pred_point_labels)

            avg_val_loss = val_total_loss / val_total_points
            logger.info(f"Epoch {epoch+1}/{cfg.scene.epochs}, Avg GNN Validation Loss: {avg_val_loss:.4f}")
            writer.add_scalar("Val/GNN_Point_Segmenter_Loss", avg_val_loss, epoch)

            overall_metrics_val = classification_metrics(stacked_true_labels_val, stacked_pred_labels_val, average='macro')
            logger.info(f"Validation Accuracy: {overall_metrics_val['accuracy']:.3f}, Validation Macro F1: {overall_metrics_val['f1']:.3f}")
            writer.add_scalar("Val/Overall_Point_Accuracy", overall_metrics_val['accuracy'], epoch)
            writer.add_scalar("Val/Overall_Point_Macro_F1", overall_metrics_val['f1'], epoch)

            all_val_preds = np.concatenate(val_pred_point_labels)
            val_pred_counts = np.bincount(all_val_preds, minlength=cfg.scene.num_classes)
            logger.info(f"Epoch {epoch+1}: Total predictions per class (val): {val_pred_counts}")
            writer.add_histogram("Val/Predictions_per_Class", val_pred_counts, epoch)

            true_counts = np.bincount(stacked_true_labels_val, minlength=cfg.scene.num_classes)
            pred_counts = np.bincount(stacked_pred_labels_val, minlength=cfg.scene.num_classes)
            
            logger.info(f"Label distribution (true, val): {true_counts}")
            logger.info(f"Prediction distribution (val): {pred_counts}")
            
            if cfg.scene.num_classes > 11:
                logger.info(f"Class 11 (static) - True: {true_counts[11]}, Predicted: {pred_counts[11]}")

            cm_all_classes = confusion_matrix(stacked_true_labels_val, stacked_pred_labels_val, 
                                              labels=[i for i in range(cfg.scene.num_classes)])
            logger.info(f"\n--- Confusion Matrix (VALIDATION, all classes) - Epoch {epoch+1} ---")
            logger.info("\n" + str(cm_all_classes))

            non_static_mask_val = stacked_true_labels_val != 11
            y_true_filtered_val = stacked_true_labels_val[non_static_mask_val]
            y_pred_filtered_val = stacked_pred_labels_val[non_static_mask_val]

            if len(y_true_filtered_val) > 0:
                labels_for_report_no_static = [i for i in range(cfg.scene.num_classes) if i != 11]
                
                logger.info(f"\n--- Classification Report (VALIDATION, excluding class 11) - Epoch {epoch+1} ---")
                report_no_static_val = classification_report(
                    y_true_filtered_val,
                    y_pred_filtered_val,
                    labels=labels_for_report_no_static,
                    zero_division=0,
                    output_dict=True 
                )
                
                logger.info(classification_report(y_true_filtered_val, y_pred_filtered_val, labels=labels_for_report_no_static, zero_division=0))

                for cls_id in labels_for_report_no_static:
                    cls_id_str = str(cls_id) 
                    if cls_id_str in report_no_static_val:
                        metrics = report_no_static_val[cls_id_str]
                        writer.add_scalar(f"Val_NoStatic/Point_Class{cls_id}_F1", metrics['f1-score'], epoch)
                        writer.add_scalar(f"Val_NoStatic/Point_Class{cls_id}_Precision", metrics['precision'], epoch)
                        writer.add_scalar(f"Val_NoStatic/Point_Class{cls_id}_Recall", metrics['recall'], epoch)
                    else:
                        logger.debug(f"Class {cls_id} not found in filtered validation report.")
            else:
                logger.warning(f"No dynamic samples for validation in epoch {epoch+1}.")

            filtered_preds_val = all_val_preds[all_val_preds != 11]
            val_pred_counts_no_static = np.bincount(filtered_preds_val, minlength=cfg.scene.num_classes - 1) 
            logger.info(f"Epoch {epoch+1}: Total predictions per class (EXCLUDING STATIC, val): {val_pred_counts_no_static}")
            writer.add_histogram("Val/Predictions_per_Class_no_static", val_pred_counts_no_static, epoch)

            per_class_f1_val = classification_metrics(stacked_true_labels_val, stacked_pred_labels_val, average=None)['f1']
            per_class_precision_val = classification_metrics(stacked_true_labels_val, stacked_pred_labels_val, average=None)['precision']
            per_class_recall_val = classification_metrics(stacked_true_labels_val, stacked_pred_labels_val, average=None)['recall']

            for cls in range(cfg.scene.num_classes):
                if cls < len(per_class_f1_val): 
                    writer.add_scalar(f"Val/Point_Class{cls}_F1", per_class_f1_val[cls], epoch)
                    writer.add_scalar(f"Val/Point_Class{cls}_Precision", per_class_precision_val[cls], epoch)
                    writer.add_scalar(f"Val/Point_Class{cls}_Recall", per_class_recall_val[cls], epoch)
        else:
            logger.warning("No points processed for validation.")

        if val_total_points > 0:
            scheduler_gnn.step(avg_val_loss)
            logger.info(f"Scheduler stepped. Current GNN LR: {optimizer_gnn.param_groups[0]['lr']:.6f}")

    writer.close()
    logger.info("HMM and GNNPointSegmenter training complete.")

if __name__ == '__main__':
    main()