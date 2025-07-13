import os
import json
import torch
import numpy as np
import hydra
from omegaconf import DictConfig
from stable_baselines3 import PPO
from torch.utils.data import DataLoader
from tqdm import tqdm 
import sys 
import logging

# Set up logging for the simulation script
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure project root is on path for imports
from hydra.utils import get_original_cwd
root = get_original_cwd()
if root not in sys.path:
    sys.path.insert(0, root)

from datasets.radar_scenes import RadarScenesDataset
from rl.env import RadarEnv 
from models.ae import ProbabilisticAE
from models.hmm import SceneTransitionModel
from models.gnn import GNNPointSegmenter 
from torch_geometric.data import Data as PyGData 
from torch_geometric.nn import knn_graph 
from torch_geometric.nn import global_mean_pool 

from utils.metrics import classification_metrics 

class CustomJsonEncoder(json.JSONEncoder):
    """Custom JSON encoder for PyTorch Tensors and NumPy objects."""
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic): 
            return obj.item()
        return super().default(obj)

def simulate(cfg: DictConfig, agent=None): 
    """
    Executes a simulation run, either with an RL agent or using default actions.
    Performs per-point segmentation and logs results.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Simulation running on device: {device}")

    ds = RadarScenesDataset(
        data_dir=cfg.data.root,
        calibration_params=cfg.calibration_params,
        num_classes=cfg.scene.num_classes 
    )

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
            logger.info(f"Loaded AE from {ae_full_path}")
        except RuntimeError as e:
            logger.error(f"CRITICAL ERROR: AE model architecture mismatch during loading. Error: {e}")
            sys.exit(1)
    else:
        raise FileNotFoundError(f"AE model not found at {ae_full_path}. Aborting simulation.")

    hmm = SceneTransitionModel(cfg.scene.num_states, cfg.sensors.latent_dim).to(device)
    hmm_full_path = os.path.join(root, cfg.scene.hmm_model_path)
    if os.path.exists(hmm_full_path):
        hmm.load_state_dict(torch.load(hmm_full_path, map_location=device))
        hmm.eval()
        logger.info(f"Loaded HMM from {hmm_full_path}")
    else:
        raise FileNotFoundError(f"HMM model not found at {hmm_full_path}. Aborting simulation.")

    gnn_point_segmenter = GNNPointSegmenter( 
        point_feature_dim=cfg.sensors.latent_dim,
        scene_context_dim=cfg.scene.num_states,
        hidden_dim=cfg.scene.hidden_dim,
        num_point_classes=cfg.scene.num_classes
    ).to(device)
    gnn_full_path = os.path.join(root, cfg.scene.gnn_model_path) 
    if os.path.exists(gnn_full_path):
        try:
            gnn_point_segmenter.load_state_dict(torch.load(gnn_full_path, map_location=device))
            gnn_point_segmenter.eval()
            logger.info(f"Loaded GNNPointSegmenter from {gnn_full_path}")
        except RuntimeError as e:
            logger.error(f"Could not load GNNPointSegmenter from {gnn_full_path}. Error: {e}. Proceeding with randomly initialized GNNPointSegmenter.")
    else:
        logger.warning(f"GNNPointSegmenter model not found at {gnn_full_path}. Proceeding with randomly initialized GNNPointSegmenter.")

    output_dir = os.path.join(os.getcwd(), cfg.sim.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, cfg.sim.log_filename)
    
    simulation_results = {"frames": []}

    all_true_points_labels_sim = []
    all_pred_points_labels_sim = []

    if agent is not None:
        logger.info("RL agent provided. Running simulation with RL policy.")
        env = RadarEnv(cfg, ds, ae=ae, hmm=hmm, gnn=gnn_point_segmenter, device=device)
        obs, info = env.reset()
    else:
        logger.info("RL agent not provided. Running simulation with default actions (no RL policy).")
        env = None
        obs = None
        info = {}

    for i in tqdm(range(len(ds)), desc="Simulating Frames"):
        if env is None:
            points, uncertainties, true_point_labels, timestamp = ds[i]
            points_tensor = points.to(device).float()
            true_point_labels_tensor = true_point_labels.to(device).long()

            with torch.no_grad():
                _, mu_per_point, _, _, _ = ae(points_tensor) 

                mu_scene = global_mean_pool(mu_per_point, torch.zeros(points_tensor.shape[0], dtype=torch.long, device=device))
                prior_states = torch.full((1, cfg.scene.num_states), 1.0/cfg.scene.num_states, device=device)
                scene_context = hmm(prior_states, mu_scene)
                
                num_neighbors = getattr(cfg.gnn, 'k_neighbors', 8)
                if points_tensor.shape[0] > 1:
                    edge_index = knn_graph(mu_per_point, k=num_neighbors, loop=False)
                else:
                    edge_index = torch.empty((2,0), dtype=torch.long, device=device)

                gnn_batch_indices = torch.zeros(mu_per_point.shape[0], dtype=torch.long, device=device)

                pred_logits = gnn_point_segmenter(
                    x=mu_per_point,
                    edge_index=edge_index,
                    scene_context=scene_context,
                    batch=gnn_batch_indices
                )
                pred_class_ids = torch.argmax(pred_logits, dim=-1)
            
            if points_tensor.numel() > 0:
                metrics_result = classification_metrics(
                    true_point_labels_tensor.cpu().numpy(),
                    pred_class_ids.cpu().numpy(),
                    average='macro'
                )
                reward = metrics_result.get('f1', 0.0)
            else:
                reward = 0.0
            
            info_step = {
                "timestamp": timestamp,
                "num_points": points_tensor.shape[0],
                "coords": points_tensor[:, :2].cpu().numpy().tolist(),
                "velocities": np.stack([points_tensor[:, 2].cpu().numpy(), np.zeros_like(points_tensor[:, 2].cpu().numpy())], axis=1).tolist(),
                "true_point_labels": true_point_labels_tensor.cpu().numpy().tolist(),
                "pred_point_labels": pred_class_ids.cpu().numpy().tolist()
            }
            action = [0.0, 0.0]
            terminated = False 
            truncated = False
        else:
            action, _ = agent.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray): 
                action = action.tolist() 
            elif isinstance(action, torch.Tensor): 
                action = action.cpu().tolist() 
            
            obs, reward, terminated, truncated, info_step = env.step(action)
        
        done = terminated or truncated

        frame_data = {
            "frame_index": i,
            "timestamp": info_step.get("timestamp", -1),
            "rl_action": action, 
            "reward": float(reward),
            "num_points": info_step.get("num_points", 0),
            "coords": info_step.get("coords", []),
            "velocities": info_step.get("velocities", []),
            "true_point_labels": info_step.get("true_point_labels", []),
            "pred_point_labels": info_step.get("pred_point_labels", [])
        }
        simulation_results["frames"].append(frame_data)

        if frame_data["num_points"] > 0:
            all_true_points_labels_sim.extend(frame_data["true_point_labels"])
            all_pred_points_labels_sim.extend(frame_data["pred_point_labels"])
        
        if done and env is not None: 
            obs, _ = env.reset()

    logger.info("\n--- Computing Overall Point Segmentation Metrics for Simulation ---")
    if len(all_true_points_labels_sim) > 0:
        stacked_true_labels_sim = np.array(all_true_points_labels_sim)
        stacked_pred_labels_sim = np.array(all_pred_points_labels_sim)

        overall_metrics_sim = classification_metrics(stacked_true_labels_sim, stacked_pred_labels_sim, average='macro')
        logger.info(f"Sim Overall Acc: {overall_metrics_sim['accuracy']:.3f}, Sim Macro F1: {overall_metrics_sim['f1']:.3f}")

        try:
            all_possible_labels = np.arange(cfg.scene.num_classes)
            target_names_full = [f'Class_{int(i)}' for i in all_possible_labels]
            
            report_sim = classification_report(
                stacked_true_labels_sim, 
                stacked_pred_labels_sim, 
                labels=all_possible_labels, 
                target_names=target_names_full, 
                zero_division=0 
            )
            logger.info("\n--- Detailed Point-Level Classification Report for Simulation ---")
            logger.info("\n" + report_sim) 

            cm_sim = confusion_matrix(stacked_true_labels_sim, stacked_pred_labels_sim, 
                                      labels=all_possible_labels)
            logger.info(f"\n--- Confusion Matrix (SIMULATION, all classes) ---")
            logger.info("\n" + str(cm_sim))

        except ValueError as e:
            logger.error(f"Could not generate classification report/confusion matrix for simulation: {e}.")

        per_class_metrics_sim = classification_metrics(stacked_true_labels_sim, stacked_pred_labels_sim, average=None)
        point_metrics = {}
        for cls_id in range(cfg.scene.num_classes):
            if cls_id in per_class_metrics_sim:
                m = per_class_metrics_sim[cls_id]
                point_metrics[str(cls_id)] = {
                    "precision": float(m['precision']),
                    "recall": float(m['recall']),
                    "f1": float(m['f1']),
                    "accuracy": float(m['accuracy']),
                    "support": int(np.sum(stacked_true_labels_sim == cls_id))
                }
            else:
                point_metrics[str(cls_id)] = {
                    "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "support": 0
                }
        simulation_results["point_metrics_overall"] = overall_metrics_sim
        simulation_results["point_metrics_per_class"] = point_metrics

    else:
        logger.warning("No valid points to compute metrics for in simulation.")

    with open(log_file_path, 'w') as f:
        json.dump(simulation_results, f, indent=2, cls=CustomJsonEncoder)
    logger.info(f"Simulation results saved to {log_file_path}")

    if env is not None: 
        env.close()

if __name__ == '__main__':
    pass