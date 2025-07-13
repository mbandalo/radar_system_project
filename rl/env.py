import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data as PyGData
from torch_geometric.nn import knn_graph, global_mean_pool

from datasets.radar_scenes import RadarScenesDataset
from models.ae import ProbabilisticAE
from models.hmm import SceneTransitionModel
from models.gnn import GNNPointSegmenter
from utils.metrics import classification_metrics

print("--- rl/env.py: Loaded with 4 dataset values unpacking. ---")

class RadarEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, cfg, dataset: RadarScenesDataset, ae=None, hmm=None, gnn=None, device=None):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.index = 0

        if ae is None or hmm is None or gnn is None or device is None:
            raise ValueError("AE, HMM, GNN models and device must be provided to RadarEnv.")
        
        self.ae = ae.eval() 
        self.hmm = hmm.eval()
        self.gnn = gnn.eval()
        self.device = device

        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.cfg.env.action_dim,), dtype=np.float32)
        
        obs_dim = self.cfg.sensors.latent_dim + self.cfg.scene.num_states + self.cfg.scene.num_classes
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(obs_dim,), dtype=np.float32)

        self._current_uncertainty = 0.0
        self._prior_dist = np.ones(self.cfg.scene.num_states, dtype=np.float32) / self.cfg.scene.num_states 
        self._posterior_dist = self._prior_dist.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        
        self._prior_dist = np.ones(self.cfg.scene.num_states, dtype=np.float32) / self.cfg.scene.num_states
        self._posterior_dist = self._prior_dist.copy()

        points, uncertainties, point_labels, timestamp = self.dataset[self.index]
        
        obs, current_scene_posterior_tensor, current_pred_point_labels = \
            self._get_observation_from_models(points, uncertainties)
        
        self._posterior_dist = current_scene_posterior_tensor.cpu().numpy().squeeze(0)

        info = {
            'timestamp': timestamp,
            'num_points': points.shape[0],
            'coords': points[:, :2].cpu().numpy().tolist(),
            'velocities': np.stack([points[:, 2].cpu().numpy(), np.zeros_like(points[:, 2].cpu().numpy())], axis=1).tolist(),
            'true_point_labels': point_labels.cpu().numpy().tolist(),
            'pred_point_labels': current_pred_point_labels.tolist(),
            'prior_dist': self._prior_dist.copy(),
            'posterior_dist': self._posterior_dist.copy()
        }
        return obs, info

    def step(self, action):
        self.index += 1
        terminated = self.index >= len(self.dataset)
        truncated = False

        reward = 0.0
        info = {}
        next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        if not terminated:
            points, uncertainties, true_point_labels, timestamp = self.dataset[self.index]

            self._prior_dist = self._posterior_dist.copy()

            next_obs, current_scene_posterior_tensor, current_pred_point_labels = \
                self._get_observation_from_models(points, uncertainties)
            
            self._posterior_dist = current_scene_posterior_tensor.cpu().numpy().squeeze(0)

            if points.numel() > 0:
                metrics_result = classification_metrics(
                    true_point_labels.cpu().numpy(),
                    current_pred_point_labels,
                    average='macro'
                )
                reward = metrics_result.get('f1', 0.0)
            else:
                reward = 0.0

            info = {
                'timestamp': timestamp,
                'num_points': points.shape[0],
                'coords': points[:, :2].cpu().numpy().tolist(),
                'velocities': np.stack([points[:, 2].cpu().numpy(), np.zeros_like(points[:, 2].cpu().numpy())], axis=1).tolist(),
                'true_point_labels': true_point_labels.cpu().numpy().tolist(),
                'pred_point_labels': current_pred_point_labels.tolist(),
                'prior_dist': self._prior_dist.copy(),
                'posterior_dist': self._posterior_dist.copy()
            }

        return next_obs, reward, terminated, truncated, info

    def _get_observation_from_models(self, points_tensor_raw: torch.Tensor, uncertainties_tensor_raw: torch.Tensor):
        """Processes raw radar data to generate RL observation and point predictions."""
        points_tensor = points_tensor_raw.float().to(self.device)

        if points_tensor.numel() == 0:
            dummy_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            dummy_posterior = torch.zeros(1, self.cfg.scene.num_states, device=self.device)
            dummy_preds = np.array([])
            return dummy_obs, dummy_posterior, dummy_preds

        with torch.no_grad():
            _, mu_per_point, _, _ = self.ae(points_tensor)
            
            mu_scene = mu_per_point.mean(dim=0, keepdim=True)
            
            hmm_prior_tensor = torch.from_numpy(self._prior_dist).float().to(self.device).unsqueeze(0)
            current_scene_posterior_tensor = self.hmm(hmm_prior_tensor, mu_scene) 

            num_neighbors = getattr(self.cfg.gnn, 'k_neighbors', 8)
            edge_index = knn_graph(mu_per_point, k=num_neighbors, loop=False)

            gnn_logits = self.gnn(
                x=mu_per_point,
                edge_index=edge_index,
                scene_context=current_scene_posterior_tensor,
                batch=None
            )

            pred_point_labels = torch.argmax(gnn_logits, dim=-1).cpu().numpy()

            observation = torch.cat([
                mu_scene.squeeze(0),
                current_scene_posterior_tensor.squeeze(0),
                F.softmax(gnn_logits, dim=-1).mean(dim=0)
            ], dim=-1).cpu().numpy()

        return observation, current_scene_posterior_tensor, pred_point_labels

    def render(self):
        pass

    def close(self):
        pass