import os
import sys
import importlib.util
import logging
from typing import Tuple, Dict, Any

import torch
import numpy as np
import hydra
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import subprocess
import collections
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import classification_report

# Import core components
try:
    from datasets.radar_scenes import RadarScenesDataset
    from models.ae import ProbabilisticAE
    from models.hmm import SceneTransitionModel
    from models.gnn import GNNPointSegmenter
    from torch_geometric.data import Data as PyGData, Batch
    from torch_geometric.nn import knn_graph, global_mean_pool
    from utils.metrics import classification_metrics
    from rl.env import RadarEnv
    from rl.agent import make_env, make_rl_model
except ImportError as e:
    print(f"Module import failed. Ensure PYTHONPATH is correctly configured. Error: {e}")
    sys.exit(1)


def collate_fn_pyg_batch(batch: Any) -> Batch:
    """Custom collate function for PyG.Batch creation."""
    data_list = []
    for points, uncertainties, point_labels, timestamp in batch:
        data_list.append(PyGData(x=points, y=point_labels, uncertainties=uncertainties, timestamp=timestamp))
    return Batch.from_data_list(data_list)


def _visualize_latent_space(
    ae_model: ProbabilisticAE,
    dataset: Dataset,
    cfg: DictConfig,
    device: torch.device,
    output_dir: str,
    logger: logging.Logger
) -> None:
    """Generates t-SNE and UMAP visualizations of the AE latent space."""
    logger.info("Collecting latent features and true labels for visualization.")

    data_loader = DataLoader(
        dataset,
        batch_size=cfg.scene.batch_size,
        shuffle=False,
        num_workers=cfg.scene.num_workers,
        collate_fn=collate_fn_pyg_batch
    )

    all_latent_features = []
    all_true_labels = []

    ae_model.eval()
    with torch.no_grad():
        for data_batch in tqdm(data_loader, desc="Extracting Latent Features for Vis."):
            data_batch = data_batch.to(device)
            if data_batch.num_nodes == 0:
                logger.warning(f"Skipping empty batch for visualization.")
                continue

            _, mu_per_point, _, _, _ = ae_model(data_batch.x)
            all_latent_features.append(mu_per_point.cpu().numpy())
            all_true_labels.append(data_batch.y.cpu().numpy())

    if not all_latent_features:
        logger.warning("No latent features collected for visualization. Skipping plot generation.")
        return

    stacked_latent_features = np.concatenate(all_latent_features, axis=0)
    stacked_true_labels = np.concatenate(all_true_labels, axis=0)

    logger.info(f"Total points collected: {len(stacked_latent_features)}")

    target_points_per_class_for_vis = getattr(cfg.scene, 'vis_points_per_class', 1000)
    sampled_latent_features_list = []
    sampled_true_labels_list = []
    unique_classes_in_data = np.unique(stacked_true_labels)

    logger.info(f"Performing stratified sampling: {target_points_per_class_for_vis} points per class.")

    for class_id in unique_classes_in_data:
        class_mask = (stacked_true_labels == class_id)
        class_features = stacked_latent_features[class_mask]
        class_labels = stacked_true_labels[class_mask]
        num_available = len(class_features)
        num_to_sample = min(target_points_per_class_for_vis, num_available)

        if num_to_sample == 0:
            logger.info(f"  Class {class_id} has no available points. Skipping.")
            continue

        indices = np.random.choice(num_available, num_to_sample, replace=False)
        sampled_latent_features_list.append(class_features[indices])
        sampled_true_labels_list.append(class_labels[indices])
        logger.info(f"  Sampled {num_to_sample} points from Class {class_id}.")

    if not sampled_latent_features_list:
        logger.warning("No points sampled after stratified sampling. Skipping plot generation.")
        return

    sampled_latent_features = np.concatenate(sampled_latent_features_list, axis=0)
    sampled_true_labels = np.concatenate(sampled_true_labels_list, axis=0)
    logger.info(f"Total points sampled for visualization: {len(sampled_latent_features)}")

    palette = sns.color_palette("hsv", cfg.scene.num_classes)

    logger.info("Performing t-SNE dimensionality reduction.")
    try:
        tsne_perplexity = min(30, len(sampled_latent_features) - 1)
        if tsne_perplexity <= 0:
            logger.warning("Not enough points for t-SNE. Skipping.")
            df_tsne = None
        else:
            tsne_results = TSNE(
                n_components=2, verbose=0, random_state=42,
                perplexity=tsne_perplexity,
                learning_rate='auto', init='random'
            ).fit_transform(sampled_latent_features)
            df_tsne = pd.DataFrame(tsne_results, columns=['tsne_dim1', 'tsne_dim2'])
            df_tsne['label'] = sampled_true_labels

        if df_tsne is not None:
            plt.figure(figsize=(12, 10))
            sns.scatterplot(
                x="tsne_dim1", y="tsne_dim2", hue="label", palette=palette,
                data=df_tsne, legend="full", alpha=0.7, s=10
            )
            plt.title('t-SNE Visualization of AE Latent Space')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plot_path_tsne = os.path.join(output_dir, "latent_space_tsne.png")
            plt.savefig(plot_path_tsne)
            logger.info(f"t-SNE plot saved to: {plot_path_tsne}")
            plt.close()
        else:
            logger.warning("t-SNE visualization skipped due to insufficient data.")
    except Exception as e:
        logger.error(f"Error during t-SNE visualization: {e}")

    logger.info("Performing UMAP dimensionality reduction.")
    try:
        if len(sampled_latent_features) < 2:
            logger.warning("Not enough points for UMAP. Skipping.")
        else:
            reducer = umap.UMAP(n_components=2, random_state=42, verbose=False)
            umap_results = reducer.fit_transform(sampled_latent_features)
            df_umap = pd.DataFrame(umap_results, columns=['umap_dim1', 'umap_dim2'])
            df_umap['label'] = sampled_true_labels

            plt.figure(figsize=(12, 10))
            sns.scatterplot(
                x="umap_dim1", y="umap_dim2", hue="label", palette=palette,
                data=df_umap, legend="full", alpha=0.7, s=10
            )
            plt.title('UMAP Visualization of AE Latent Space')
            plt.xlabel('UMAP Dimension 1')
            plt.ylabel('UMAP Dimension 2')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plot_path_umap = os.path.join(output_dir, "latent_space_umap.png")
            plt.savefig(plot_path_umap)
            logger.info(f"UMAP plot saved to: {plot_path_umap}")
            plt.close()
    except Exception as e:
        logger.error(f"Error during UMAP visualization: {e}")

    logger.info("Latent space visualization complete.")


def train_or_check_models(
    script_name: str,
    model_paths_relative_to_root: Tuple[str, ...],
    force_retrain_flag_name: str,
    cfg: DictConfig,
    root: str,
    logger: logging.Logger
) -> None:
    """Ensures models are trained or loaded based on configuration."""
    abs_model_paths = [os.path.join(root, p) for p in model_paths_relative_to_root]

    force_retrain = False
    if "sensor" in script_name:
        force_retrain = getattr(cfg.sensors, force_retrain_flag_name, False)
    elif "scene" in script_name:
        force_retrain = getattr(cfg.scene, force_retrain_flag_name, False)
    elif "sim" in script_name:
        force_retrain = getattr(cfg.sim, force_retrain_flag_name, False)

    all_exist = all(os.path.exists(p) for p in abs_model_paths)

    if force_retrain or not all_exist:
        action = "forcing retraining" if force_retrain else "training missing models"
        logger.info(f"Models: {[os.path.relpath(p, root) for p in abs_model_paths]} - {action} via {script_name}.")

        if force_retrain:
            for p in abs_model_paths:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                        logger.info(f"Deleted old checkpoint: {p}")
                    except OSError as e:
                        logger.warning(f"Could not delete {p}: {e}. Skipping.")

        env = os.environ.copy()
        env['PYTHONPATH'] = root + os.pathsep + env.get('PYTHONPATH', '')

        try:
            subprocess.run(
                [sys.executable, os.path.join(root, script_name)],
                check=True,
                cwd=root,
                env=env,
                capture_output=True,
                text=True
            )
            logger.info(f"Training for {script_name} complete.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error training {script_name}: {e}\nStdout:\n{e.stdout}\nStderr:\n{e.stderr}")
            sys.exit(1)
        except FileNotFoundError:
            logger.error(f"Training script not found: {os.path.join(root, script_name)}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"An unexpected error occurred during training {script_name}: {e}")
            sys.exit(1)
    else:
        logger.info(f"All models for {script_name} found, skipping training.")


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the end-to-end radar system. Orchestrates model management,
    evaluation, and simulation.
    """
    from hydra.utils import get_original_cwd
    project_root = get_original_cwd()
    sys.path.insert(0, project_root)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Initializing main application execution.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    current_run_output_dir = os.getcwd()
    writer = SummaryWriter(os.path.join(current_run_output_dir, 'tensorboard_main_metrics'))
    logger.info(f"TensorBoard logs: {os.path.join(current_run_output_dir, 'tensorboard_main_metrics')}")

    ae_model_path = cfg.sensors.ae_model_path
    hmm_model_path = cfg.scene.hmm_model_path
    gnn_model_path = cfg.scene.gnn_model_path
    rl_model_name = cfg.sim.rl_model_name

    logger.info("\n--- Checking/Training Models ---")
    train_or_check_models(
        script_name="training/train_sensor_module.py",
        model_paths_relative_to_root=(ae_model_path,),
        force_retrain_flag_name='force_retrain_ae',
        cfg=cfg, root=project_root, logger=logger
    )
    train_or_check_models(
        script_name="training/train_scene_module.py",
        model_paths_relative_to_root=(hmm_model_path, gnn_model_path),
        force_retrain_flag_name='force_retrain_scene',
        cfg=cfg, root=project_root, logger=logger
    )
    train_or_check_models(
         script_name="training/train_system.py",
         model_paths_relative_to_root=(rl_model_name,),
         force_retrain_flag_name='force_retrain_rl',
         cfg=cfg, root=project_root, logger=logger
     )

    logger.info("\n--- Loading Dataset ---")
    try:
        ds = RadarScenesDataset(
            data_dir=cfg.data.root,
            calibration_params=cfg.calibration_params,
            num_classes=cfg.scene.num_classes
        )
        train_size = int(len(ds) * cfg.data.train_frac)
        val_size = len(ds) - train_size
        train_ds, val_ds = random_split(ds, [train_size, val_size])
        logger.info(f"Dataset split: Train = {len(train_ds)}, Val = {len(val_ds)}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)

    logger.info("\n--- Loading Trained Models ---")
    ae = ProbabilisticAE(
        input_dim=cfg.sensors.input_dim, latent_dim=cfg.sensors.latent_dim,
        hidden_dims=cfg.sensors.hidden_dims, num_classes=cfg.scene.num_classes
    ).to(device)
    ae_full_path = os.path.join(project_root, ae_model_path)
    if os.path.exists(ae_full_path):
        try:
            ae.load_state_dict(torch.load(ae_full_path, map_location=device))
            ae.eval()
            logger.info(f"Loaded AE from {ae_full_path}")
        except Exception as e:
            logger.error(f"Failed to load AE model: {e}")
            sys.exit(1)
    else:
        logger.error(f"AE model not found: {ae_full_path}")
        sys.exit(1)

    logger.info("\n--- Generating AE Latent Space Visualization ---")
    _visualize_latent_space(ae, val_ds, cfg, device, current_run_output_dir, logger)

    hmm = SceneTransitionModel(cfg.scene.num_states, cfg.sensors.latent_dim).to(device)
    hmm_full_path = os.path.join(project_root, hmm_model_path)
    if os.path.exists(hmm_full_path):
        try:
            hmm.load_state_dict(torch.load(hmm_full_path, map_location=device))
            hmm.eval()
            logger.info(f"Loaded HMM from {hmm_full_path}")
        except Exception as e:
            logger.warning(f"Failed to load HMM model: {e}")

    point_segmenter_gnn = GNNPointSegmenter(
        point_feature_dim=cfg.sensors.latent_dim, scene_context_dim=cfg.scene.num_states,
        hidden_dim=cfg.scene.hidden_dim, num_point_classes=cfg.scene.num_classes
    ).to(device)
    gnn_full_path = os.path.join(project_root, gnn_model_path)
    if os.path.exists(gnn_full_path):
        try:
            point_segmenter_gnn.load_state_dict(torch.load(gnn_full_path, map_location=device))
            point_segmenter_gnn.eval()
            logger.info(f"Loaded GNNPointSegmenter from {gnn_full_path}")
        except RuntimeError as e:
            logger.warning(f"Could not load GNNPointSegmenter: {e}. Randomly initialized.")
    else:
        logger.warning(f"GNNPointSegmenter model not found: {gnn_full_path}. Randomly initialized.")

    agent = None
    rl_model_full_path = os.path.join(project_root, rl_model_name)
    if os.path.exists(rl_model_full_path):
        try:
            if 'make_env' in sys.modules and 'PPO' in sys.modules['stable_baselines3']:
                envs = make_env(cfg, train_ds, ae=ae, hmm=hmm, gnn=point_segmenter_gnn, device=device)
                from stable_baselines3 import PPO
                agent = PPO.load(rl_model_full_path, env=envs)
                logger.info(f"Loaded PPO agent from {rl_model_full_path}")
            else:
                logger.warning("RL components unavailable. Cannot load RL agent.")
        except Exception as e:
            logger.warning(f"Failed to load PPO agent: {e}. Skipping RL agent.")
    else:
        logger.info("RL agent model not found. Skipping RL agent loading.")

    logger.info("\n--- Evaluating Per-Point Segmentation ---")
    val_loader = DataLoader(
        val_ds, batch_size=cfg.scene.batch_size, shuffle=False,
        num_workers=cfg.scene.num_workers, collate_fn=collate_fn_pyg_batch
    )
    all_true_point_labels = []
    all_pred_point_labels = []

    ae.eval()
    hmm.eval()
    point_segmenter_gnn.eval()

    with torch.no_grad():
        for data_batch in tqdm(val_loader, desc="Evaluating Per-Point Segmentation"):
            data_batch = data_batch.to(device)
            if data_batch.num_nodes == 0:
                logger.warning(f"Skipping empty batch during evaluation.")
                continue

            _, mu_per_point, _, _, _ = ae(data_batch.x)
            mu_scene_batch = global_mean_pool(mu_per_point, data_batch.batch)
            prior_states = torch.full((mu_scene_batch.size(0), cfg.scene.num_states), 1.0/cfg.scene.num_states, device=device)
            scene_context = hmm(prior_states, mu_scene_batch)
            num_neighbors = getattr(cfg.gnn, 'k_neighbors', 8)
            edge_index = knn_graph(mu_per_point, k=num_neighbors, batch=data_batch.batch, loop=False)

            pred_logits = point_segmenter_gnn(
                x=mu_per_point, edge_index=edge_index,
                scene_context=scene_context, batch=data_batch.batch
            )
            pred_class_ids = torch.argmax(pred_logits, dim=-1).cpu().numpy()
            all_true_point_labels.append(data_batch.y.cpu().numpy())
            all_pred_point_labels.append(pred_class_ids)

    if len(all_true_point_labels) > 0:
        stacked_true_labels = np.concatenate(all_true_point_labels)
        stacked_pred_labels = np.concatenate(all_pred_point_labels)

        logger.info("\n--- Overall Point Segmentation Metrics ---")
        overall_metrics = classification_metrics(stacked_true_labels, stacked_pred_labels, average='macro')
        logger.info(f"Val Overall Acc: {overall_metrics['accuracy']:.3f}, Macro F1: {overall_metrics['f1']:.3f}")
        writer.add_scalar("Eval/Overall_Point_Accuracy", overall_metrics['accuracy'], 0)
        writer.add_scalar("Eval/Overall_Point_Macro_F1", overall_metrics['f1'], 0)

        logger.info("\n--- Detailed Point-Level Classification Report ---")
        try:
            all_possible_labels = np.arange(cfg.scene.num_classes)
            target_names_full = [f'Class_{int(i)}' for i in all_possible_labels]
            report = classification_report(
                stacked_true_labels, stacked_pred_labels,
                labels=all_possible_labels, target_names=target_names_full,
                zero_division=0
            )
            logger.info("\n" + report)
        except ValueError as e:
            logger.error(f"Failed to generate classification report: {e}")
    else:
        logger.warning("No valid points processed for evaluation.")

    logger.info("\n--- Running Simulation ---")
    try:
        spec = importlib.util.spec_from_file_location('simulate', os.path.join(project_root, 'training', 'simulate.py'))
        sim_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sim_mod)
        sim_mod.simulate(cfg, agent=agent)
        logger.info("Simulation complete.")
    except Exception as e:
        logger.error(f"Error during simulation: {e}")

    writer.close()
    logger.info("Main execution finished.")

if __name__ == '__main__':
    main()