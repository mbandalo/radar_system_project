import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import hydra
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging 
import torch.nn as nn

import numpy as np

from datasets.radar_scenes import RadarScenesDataset 
from models.ae import ProbabilisticAE, gaussian_nll_loss, kl_divergence 

print("--- training/train_sensor_module.py: Initializing AE training. ---")

def collate_fn_pad(batch):
    """Custom collate function for batching and padding radar points."""
    if not batch:
        return torch.empty(0, 7), torch.empty(0, 1), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    max_points = 0
    valid_batch_items = [] 
    for points, uncertainties, point_labels, timestamp in batch: 
        if points.size(0) > 0:
            valid_batch_items.append((points, uncertainties, point_labels, timestamp))
            if points.size(0) > max_points:
                max_points = points.size(0)
    
    if not valid_batch_items:
        # Return empty tensors with correct dimensions if all items were empty
        return torch.empty(0, batch[0][0].size(1)), \
               torch.empty(0, batch[0][1].size(1) if batch[0][1].dim() > 1 else 1), \
               torch.empty(0, dtype=torch.long), \
               torch.empty(0, dtype=torch.long)

    points_list = []
    uncertainties_list = []
    point_labels_list = []
    timestamps_list = []

    for points, uncertainties, point_labels, timestamp in valid_batch_items:
        num_points = points.size(0)
        num_features = points.size(1)
        padded_points = torch.zeros(max_points, num_features, dtype=points.dtype)
        padded_points[:num_points, :] = points
        points_list.append(padded_points)

        if uncertainties.dim() == 1:
            padded_uncertainties = torch.zeros(max_points, 1, dtype=uncertainties.dtype)
            padded_uncertainties[:num_points, 0] = uncertainties
        else: 
            num_unc_features = uncertainties.size(1)
            padded_uncertainties = torch.zeros(max_points, num_unc_features, dtype=uncertainties.dtype)
            padded_uncertainties[:num_points, :] = uncertainties
        uncertainties_list.append(padded_uncertainties)

        padded_point_labels = torch.full((max_points,), -1, dtype=point_labels.dtype)
        padded_point_labels[:num_points] = point_labels
        point_labels_list.append(padded_point_labels)

        timestamps_list.append(torch.tensor(timestamp, dtype=torch.long))

    batch_points = torch.stack(points_list, dim=0)
    batch_uncertainties = torch.stack(uncertainties_list, dim=0)
    batch_point_labels = torch.stack(point_labels_list, dim=0)
    batch_timestamps = torch.stack(timestamps_list, dim=0)
    
    return batch_points, batch_uncertainties, batch_point_labels, batch_timestamps 

@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    from hydra.utils import get_original_cwd 
    project_root = get_original_cwd() 
    if project_root not in sys.path:
        sys.path.insert(0, project_root) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.autograd.set_detect_anomaly(True) 
    
    log_output_dir = os.path.join(os.getcwd(), 'logs', 'ae_training') 
    os.makedirs(log_output_dir, exist_ok=True)
    debug_log_path = os.path.join(log_output_dir, "ae_debug_log.txt")

    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(debug_log_path, mode='w'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    logger = logging.getLogger(__name__) 
    logger.info(f"AE training diagnostics logged to: {debug_log_path}")

    log_dir_base = os.path.join(os.getcwd(), 'tensorboard_sensor')
    writer = SummaryWriter(log_dir_base)
    logger.info(f"TensorBoard (sensor): {log_dir_base}")
    
    dataset = RadarScenesDataset(
        data_dir=cfg.data.root,
        calibration_params=cfg.calibration_params,
        num_classes=cfg.scene.num_classes 
    )
    train_size = int(cfg.data.train_frac * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    logger.info(f"Dataset split: Train = {len(train_dataset)}, Val = {len(val_dataset)}")

    logger.info("Calculating class weights for AE reconstruction loss.")
    all_train_point_labels_flat = []
    temp_train_loader_flat = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=cfg.sensors.num_workers, collate_fn=collate_fn_pad)
    
    for i, (points, uncertainties, point_labels_batch_padded, timestamp) in enumerate(tqdm(temp_train_loader_flat, desc="Collecting AE labels")):
        current_labels_flat = point_labels_batch_padded.flatten()
        valid_labels = current_labels_flat[current_labels_flat != -1]
        if valid_labels.numel() > 0:
            all_train_point_labels_flat.append(valid_labels.cpu().numpy())

    if not all_train_point_labels_flat:
        logger.warning("No valid point labels collected. Class weights will be uniform.")
        class_weights_for_ae_loss = torch.ones(cfg.scene.num_classes).to(device)
    else:
        stacked_train_labels_flat = np.concatenate(all_train_point_labels_flat)
        class_counts = np.bincount(stacked_train_labels_flat, minlength=cfg.scene.num_classes)
        
        epsilon = 1e-8 
        class_counts_tensor = torch.tensor(class_counts, dtype=torch.float32)
        
        total_points_in_dataset = class_counts_tensor.sum()
        num_classes_in_cfg = cfg.scene.num_classes
        
        class_weights_for_ae_loss = torch.zeros(num_classes_in_cfg, dtype=torch.float32)
        for c in range(num_classes_in_cfg):
            if class_counts[c] > 0:
                class_weights_for_ae_loss[c] = total_points_in_dataset / (num_classes_in_cfg * class_counts_tensor[c])
            else:
                class_weights_for_ae_loss[c] = epsilon 
        
        if hasattr(cfg.scene, 'max_class_weight'): 
            class_weights_for_ae_loss = torch.clamp(class_weights_for_ae_loss, max=cfg.scene.max_class_weight)
        
        class_weights_for_ae_loss = class_weights_for_ae_loss.to(device)
        
        logger.info(f"AE Class counts: {class_counts}")
        logger.info(f"Calculated AE class weights: {class_weights_for_ae_loss.cpu().numpy()}")

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.sensors.batch_size,
                              shuffle=True, 
                              num_workers=cfg.sensors.num_workers,
                              collate_fn=collate_fn_pad)

    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.sensors.batch_size,
                            shuffle=False, 
                            num_workers=cfg.sensors.num_workers,
                            collate_fn=collate_fn_pad)

    ae = ProbabilisticAE(
        input_dim=cfg.sensors.input_dim,
        latent_dim=cfg.sensors.latent_dim,
        hidden_dims=cfg.sensors.hidden_dims,
        num_classes=cfg.scene.num_classes 
    ).to(device)

    optimizer = optim.Adam(ae.parameters(), lr=cfg.sensors.lr)

    class_criterion_ae = nn.CrossEntropyLoss(reduction='mean') 

    logger.info(f"\n--- Training AE for {cfg.sensors.epochs} epochs ---")
    for epoch in range(cfg.sensors.epochs):
        ae.train()
        total_loss_epoch = 0.0
        total_recon_loss_epoch = 0.0
        total_kl_loss_epoch = 0.0
        total_class_loss_epoch = 0.0 
        num_valid_batches_train = 0 

        for i, (points_padded, uncertainties_padded, point_labels_padded, timestamps) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training AE")):
            points_padded = points_padded.to(device) 
            point_labels_padded = point_labels_padded.to(device) 

            valid_mask_batch = (point_labels_padded != -1)
            
            points_flat_unfiltered = points_padded.view(-1, points_padded.size(-1))
            labels_flat_unfiltered = point_labels_padded.view(-1)
            
            valid_mask = valid_mask_batch.view(-1) 

            points_flat = points_flat_unfiltered[valid_mask]
            point_labels_flat = labels_flat_unfiltered[valid_mask]

            if not valid_mask.any() or points_flat.numel() == 0: 
                logger.warning(f"Batch {i+1} has no valid points. Skipping.")
                continue

            num_valid_batches_train += 1 

            optimizer.zero_grad()
            
            recon_mean_flat, mu_flat, log_var_flat, x_rec_logvar_flat, classification_logits_flat = ae(points_flat) 

            if recon_mean_flat.numel() > 0: 
                recon_loss_per_point_unweighted = gaussian_nll_loss(recon_mean_flat, points_flat, x_rec_logvar_flat, reduction='none')
                
                if point_labels_flat.max() >= class_weights_for_ae_loss.size(0) or point_labels_flat.min() < 0:
                    logger.error(f"AE Recon Loss: Labels out of range for class weights: Max={point_labels_flat.max()}, Min={point_labels_flat.min()}")
                    continue 

                weights_for_current_points = class_weights_for_ae_loss[point_labels_flat]
                weighted_recon_loss = (recon_loss_per_point_unweighted * weights_for_current_points).mean()
            else:
                weighted_recon_loss = torch.tensor(0.0, device=device)
            
            kl_loss_value = kl_divergence(mu_flat, log_var_flat, free_bits=cfg.sensors.free_bits) 

            classification_loss = torch.tensor(0.0, device=device)
            if ae.classifier is not None and classification_logits_flat.numel() > 0 and point_labels_flat.numel() > 0:
                if point_labels_flat.max() >= ae.num_classes or point_labels_flat.min() < 0:
                    logger.error(f"AE Class Loss: Labels out of range for classifier: Max={point_labels_flat.max()}, Min={point_labels_flat.min()}")
                    continue 
                
                classification_loss = class_criterion_ae(classification_logits_flat, point_labels_flat)
            
            total_loss_current_batch = weighted_recon_loss + cfg.sensors.beta * kl_loss_value + \
                                       (classification_loss * getattr(cfg.sensors, 'ae_classification_loss_weight', 0.1)) 

            if torch.isnan(total_loss_current_batch) or torch.isinf(total_loss_current_batch):
                logger.error(f"AE Training: NaN/Inf Total Loss in Epoch {epoch+1}, Batch {i+1}. Skipping batch.")
                continue 

            total_loss_current_batch.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=getattr(cfg.sensors, 'clip_grad_norm_ae', 1.0)) 
            optimizer.step()

            total_loss_epoch += total_loss_current_batch.item()
            total_recon_loss_epoch += weighted_recon_loss.item()
            total_kl_loss_epoch += kl_loss_value.item() 
            total_class_loss_epoch += classification_loss.item() 

            if (i + 1) % getattr(cfg.sensors, 'log_interval_ae', 5) == 0 or i == len(train_loader) - 1:
                logger.info(f"AE Diag Batch {i+1}:")
                logger.info(f"  Input Points (Min/Max/Mean): {points_flat.min().item():.6f}/{points_flat.max().item():.6f}/{points_flat.mean().item():.6f}")
                logger.info(f"  Mu Latent (Min/Max/Mean): {mu_flat.min().item():.6f}/{mu_flat.max().item():.6f}/{mu_flat.mean().item():.6f}")
                logger.info(f"  LogVar Latent (Min/Max/Mean): {log_var_flat.min().item():.6f}/{log_var_flat.max().item():.6f}/{log_var_flat.mean().item():.6f}")
                logger.info(f"  Recon Mean (Min/Max/Mean): {recon_mean_flat.min().item():.6f}/{recon_mean_flat.max().item():.6f}/{recon_mean_flat.mean().item():.6f}")
                logger.info(f"  Recon LogVar (Min/Max/Mean): {x_rec_logvar_flat.min().item():.6f}/{x_rec_logvar_flat.max().item():.6f}/{x_rec_logvar_flat.mean().item():.6f}")
                logger.info(f"  Weighted Recon Loss: {weighted_recon_loss.item():.6f}, KL Loss: {kl_loss_value.item():.6f}, Class Loss: {classification_loss.item():.6f}, Total Loss: {total_loss_current_batch.item():.6f}")

                if hasattr(ae.encoder, 'net') and len(ae.encoder.net) > 0 and ae.encoder.net[0].weight.grad is not None: 
                    grad_norm = ae.encoder.net[0].weight.grad.norm().item()
                    logger.info(f"  Encoder First Layer Grad Norm: {grad_norm:.6f}")
                else:
                    logger.info("  Encoder First Layer Grad unavailable.")


        if num_valid_batches_train > 0: 
            avg_loss = total_loss_epoch / num_valid_batches_train
            avg_recon_loss = total_recon_loss_epoch / num_valid_batches_train
            avg_kl_loss = total_kl_loss_epoch / num_valid_batches_train
            avg_class_loss = total_class_loss_epoch / num_valid_batches_train 
            logger.info(f"Epoch {epoch+1}/{cfg.sensors.epochs}, AE Loss: {avg_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}, Class Loss: {avg_class_loss:.4f}")
            writer.add_scalar('Loss/AE_Train', avg_loss, epoch)
            writer.add_scalar('Loss/AE_Recon_Train', avg_recon_loss, epoch)
            writer.add_scalar('Loss/AE_KL_Train', avg_kl_loss, epoch)
            writer.add_scalar('Loss/AE_Class_Train', avg_class_loss, epoch) 
        else:
            logger.warning(f"Epoch {epoch+1}: No valid batches processed in training.")

        ae.eval()
        val_total_loss_epoch = 0.0
        val_total_recon_loss_epoch = 0.0
        val_total_kl_loss_epoch = 0.0
        val_total_class_loss_epoch = 0.0 
        val_num_valid_batches = 0 

        with torch.no_grad():
            for i, (points_padded, uncertainties_padded, point_labels_padded, timestamps) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} Validation AE")):
                points_padded = points_padded.to(device)
                point_labels_padded = point_labels_padded.to(device)
                
                valid_mask_batch = (point_labels_padded != -1)
                points_flat_unfiltered = points_padded.view(-1, points_padded.size(-1))
                labels_flat_unfiltered = point_labels_padded.view(-1)
                valid_mask = valid_mask_batch.view(-1)

                points_flat = points_flat_unfiltered[valid_mask]
                point_labels_flat = labels_flat_unfiltered[valid_mask]

                if not valid_mask.any() or points_flat.numel() == 0:
                    continue
                
                val_num_valid_batches += 1

                recon_mean_flat, mu_flat, log_var_flat, x_rec_logvar_flat, classification_logits_flat = ae(points_flat) 

                if recon_mean_flat.numel() > 0:
                    recon_loss_per_point_unweighted = gaussian_nll_loss(recon_mean_flat, points_flat, x_rec_logvar_flat, reduction='none')
                    
                    if point_labels_flat.max() >= class_weights_for_ae_loss.size(0) or point_labels_flat.min() < 0:
                        logger.error(f"AE Recon Val Loss: Labels out of range for class weights: Max={point_labels_flat.max()}, Min={point_labels_flat.min()}")
                        continue 

                    weights_for_current_points = class_weights_for_ae_loss[point_labels_flat]
                    weighted_recon_loss = (recon_loss_per_point_unweighted * weights_for_current_points).mean()
                else:
                    weighted_recon_loss = torch.tensor(0.0, device=device)
                
                kl_loss_value = kl_divergence(mu_flat, log_var_flat, free_bits=cfg.sensors.free_bits)
                
                classification_val_loss = torch.tensor(0.0, device=device)
                if ae.classifier is not None and classification_logits_flat.numel() > 0 and point_labels_flat.numel() > 0:
                    if point_labels_flat.max() >= ae.num_classes or point_labels_flat.min() < 0:
                        logger.error(f"AE Class Val Loss: Labels out of range for classifier: Max={point_labels_flat.max()}, Min={point_labels_flat.min()}")
                        continue 
                    classification_val_loss = class_criterion_ae(classification_logits_flat, point_labels_flat)

                loss = weighted_recon_loss + cfg.sensors.beta * kl_loss_value + \
                       (classification_val_loss * getattr(cfg.sensors, 'ae_classification_loss_weight', 0.1))

                val_total_loss_epoch += loss.item()
                val_total_recon_loss_epoch += weighted_recon_loss.item()
                val_total_kl_loss_epoch += kl_loss_value.item()
                val_total_class_loss_epoch += classification_val_loss.item() 
        
        if val_num_valid_batches > 0:
            avg_val_loss = val_total_loss_epoch / val_num_valid_batches
            avg_val_recon_loss = val_total_recon_loss_epoch / val_num_valid_batches
            avg_val_kl_loss = val_total_kl_loss_epoch / val_num_valid_batches
            avg_val_class_loss = val_total_class_loss_epoch / val_num_valid_batches 
            logger.info(f"Epoch {epoch+1}/{cfg.sensors.epochs}, AE Val Loss: {avg_val_loss:.4f}, Recon Val Loss: {avg_val_recon_loss:.4f}, KL Val Loss: {avg_val_kl_loss:.4f}, Class Val Loss: {avg_val_class_loss:.4f}")
            writer.add_scalar('Loss/AE_Val', avg_val_loss, epoch)
            writer.add_scalar('Loss/AE_Recon_Val', avg_val_recon_loss, epoch)
            writer.add_scalar('Loss/AE_KL_Val', avg_val_kl_loss, epoch)
            writer.add_scalar('Loss/AE_Class_Val', avg_val_class_loss, epoch) 
        else:
            logger.warning(f"Epoch {epoch+1}: No valid batches processed in validation.")

    ae_save_path_full = os.path.join(project_root, cfg.sensors.ae_model_path)
    os.makedirs(os.path.dirname(ae_save_path_full), exist_ok=True)
    torch.save(ae.state_dict(), ae_save_path_full)
    logger.info(f"Saved AE to {ae_save_path_full}")

    writer.close()
    logger.info("Sensor module training complete.")

if __name__ == '__main__':
    main()