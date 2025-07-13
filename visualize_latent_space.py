import os
import torch
import numpy as np
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from datasets.radar_scenes import RadarScenesDataset
from models.ae import ProbabilisticAE
from torch_geometric.data import Data as PyGData, Batch

def collate_fn_pyg_batch(batch):
    """Custom collate function for PyG.Batch creation."""
    data_list = []
    for points, uncertainties, point_labels, timestamp in batch:
        data_list.append(PyGData(x=points, y=point_labels, uncertainties=uncertainties, timestamp=timestamp))
    return Batch.from_data_list(data_list)

@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def visualize_latent_space(cfg: DictConfig):
    """Generates t-SNE and UMAP visualizations of the AE latent space."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    from hydra.utils import get_original_cwd
    root = get_original_cwd()

    ds = RadarScenesDataset(
        data_dir=cfg.data.root,
        calibration_params=cfg.calibration_params,
        num_classes=cfg.scene.num_classes
    )
    _, val_ds = random_split(ds, [int(len(ds) * cfg.data.train_frac), len(ds) - int(len(ds) * cfg.data.train_frac)])
    logger.info(f"Loaded {len(val_ds)} validation samples for visualization.")

    val_loader = DataLoader(val_ds, batch_size=cfg.scene.batch_size, shuffle=False,
                            num_workers=cfg.scene.num_workers, collate_fn=collate_fn_pyg_batch)

    ae = ProbabilisticAE(
        input_dim=cfg.sensors.input_dim,
        latent_dim=cfg.sensors.latent_dim,
        hidden_dims=cfg.sensors.hidden_dims
    ).to(device)

    ae_full_path = os.path.join(root, cfg.sensors.ae_model_path)

    if os.path.exists(ae_full_path):
        ae.load_state_dict(torch.load(ae_full_path, map_location=device))
        ae.eval()
        logger.info(f"Loaded AE checkpoint from: {ae_full_path}")
    else:
        logger.error(f"AE model not found at '{ae_full_path}'. Visualization aborted.")
        return

    logger.info("Collecting latent features and true labels.")
    all_latent_features = []
    all_true_labels = []

    with torch.no_grad():
        for batch_idx, data_batch in enumerate(tqdm(val_loader, desc="Extracting Latent Features")):
            data_batch = data_batch.to(device)

            if data_batch.num_nodes == 0:
                logger.warning(f"Batch {batch_idx+1} is empty. Skipping.")
                continue

            _, mu_per_point, _, _ = ae(data_batch.x)
            all_latent_features.append(mu_per_point.cpu().numpy())
            all_true_labels.append(data_batch.y.cpu().numpy())

    if not all_latent_features:
        logger.error("No latent features collected. Check data or model output.")
        return

    stacked_latent_features = np.concatenate(all_latent_features, axis=0)
    stacked_true_labels = np.concatenate(all_true_labels, axis=0)

    logger.info(f"Total points collected: {len(stacked_latent_features)}")
    logger.info(f"Shape of collected latent features: {stacked_latent_features.shape}")

    logger.info("Performing dimensionality reduction (t-SNE/UMAP).")

    try:
        tsne_results = TSNE(n_components=2, verbose=1, random_state=42, perplexity=30, n_iter=1000).fit_transform(stacked_latent_features)
        df_tsne = pd.DataFrame(tsne_results, columns=['tsne_dim1', 'tsne_dim2'])
        df_tsne['label'] = stacked_true_labels

        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x="tsne_dim1", y="tsne_dim2",
            hue="label",
            palette=sns.color_palette("hsv", cfg.scene.num_classes),
            data=df_tsne,
            legend="full",
            alpha=0.7,
            s=10
        )
        plt.title('t-SNE Visualization of AE Latent Space')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plot_path_tsne = os.path.join(root, cfg.sim.output_dir, "latent_space_tsne.png")
        plt.savefig(plot_path_tsne)
        logger.info(f"t-SNE plot saved to: {plot_path_tsne}")
        plt.close()
    except Exception as e:
        logger.error(f"Error during t-SNE visualization: {e}")

    try:
        reducer = umap.UMAP(n_components=2, random_state=42, verbose=False)
        umap_results = reducer.fit_transform(stacked_latent_features)
        df_umap = pd.DataFrame(umap_results, columns=['umap_dim1', 'umap_dim2'])
        df_umap['label'] = stacked_true_labels

        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x="umap_dim1", y="umap_dim2",
            hue="label",
            palette=sns.color_palette("hsv", cfg.scene.num_classes),
            data=df_umap,
            legend="full",
            alpha=0.7,
            s=10
        )
        plt.title('UMAP Visualization of AE Latent Space')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plot_path_umap = os.path.join(root, cfg.sim.output_dir, "latent_space_umap.png")
        plt.savefig(plot_path_umap)
        logger.info(f"UMAP plot saved to: {plot_path_umap}")
        plt.close()
    except Exception as e:
        logger.error(f"Error during UMAP visualization: {e}")

    logger.info("Latent space visualization complete.")

if __name__ == '__main__':
    visualize_latent_space()