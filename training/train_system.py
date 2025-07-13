import os
import sys
import torch
import numpy as np
import hydra
from omegaconf import DictConfig
from stable_baselines3 import PPO
from torch.utils.data import DataLoader 

from hydra.utils import get_original_cwd

from datasets.radar_scenes import RadarScenesDataset
from rl.env import RadarEnv 
from rl.agent import make_env, make_rl_model 
from models.ae import ProbabilisticAE
from models.hmm import SceneTransitionModel
from models.gnn import GNNPointSegmenter 
from torch_geometric.data import Data as PyGData 
from torch_geometric.nn import knn_graph 
from torch_geometric.nn import global_mean_pool 

@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    """
    Trains the Reinforcement Learning (RL) agent (PPO) using the RadarEnv.
    This script loads the pre-trained AE, HMM, and GNNPointSegmenter models.
    """
    root = get_original_cwd()
    sys.path.insert(0, root) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"RL training running on device: {device}")

    ds = RadarScenesDataset(
        data_dir=cfg.data.root,
        calibration_params=cfg.calibration_params,
        num_classes=cfg.scene.num_classes 
    )
    split = int(len(ds) * cfg.data.train_frac)
    train_ds, _ = torch.utils.data.random_split(ds, [split, len(ds)-split])
    print(f"RL environment using {len(train_ds)} training samples.")


    # 1. Load pre-trained AE model
    ae = ProbabilisticAE(
        input_dim=cfg.sensors.input_dim,
        latent_dim=cfg.sensors.latent_dim,
        hidden_dims=cfg.sensors.hidden_dims
    ).to(device)
    ae_full_path = os.path.join(root, cfg.sensors.ae_model_path)
    if os.path.exists(ae_full_path):
        ae.load_state_dict(torch.load(ae_full_path, map_location=device))
        ae.eval() 
        print(f"Loaded AE from {ae_full_path}")
    else:
        raise FileNotFoundError(f"AE model not found at {ae_full_path}. Please train it first.")

    # 2. Load pre-trained HMM model
    hmm = SceneTransitionModel(cfg.scene.num_states, cfg.sensors.latent_dim).to(device)
    hmm_full_path = os.path.join(root, cfg.scene.hmm_model_path)
    if os.path.exists(hmm_full_path):
        hmm.load_state_dict(torch.load(hmm_full_path, map_location=device))
        hmm.eval()
        print(f"Loaded HMM from {hmm_full_path}")
    else:
        raise FileNotFoundError(f"HMM model not found at {hmm_full_path}. Please train it first.")

    # 3. Load pre-trained GNNPointSegmenter model
    gnn = GNNPointSegmenter( 
        point_feature_dim=cfg.sensors.latent_dim,
        scene_context_dim=cfg.scene.num_states,
        hidden_dim=cfg.scene.hidden_dim,
        num_point_classes=cfg.scene.num_classes
    ).to(device)
    gnn_full_path = os.path.join(root, cfg.scene.gnn_model_path) 
    if os.path.exists(gnn_full_path):
        gnn.load_state_dict(torch.load(gnn_full_path, map_location=device))
        gnn.eval() 
        print(f"Loaded GNNPointSegmenter from {gnn_full_path}")
    else:
        raise FileNotFoundError(f"GNNPointSegmenter model not found at {gnn_full_path}. Please train it first.")

    # 4. Create RL environment(s)
    envs = make_env(cfg, train_ds, ae=ae, hmm=hmm, gnn=gnn, device=device)
    print(f"Created {cfg.rl.n_envs} parallel environments.")

    # 5. Initialize or load PPO agent
    rl_model_full_path = os.path.join(root, cfg.sim.rl_model_name)
    if os.path.exists(rl_model_full_path):
        agent = PPO.load(rl_model_full_path, env=envs, device=device)
        print(f"Loaded PPO agent from {rl_model_full_path}")
    else:
        agent = PPO(
            cfg.rl.policy,
            envs,
            verbose=1,
            learning_rate=cfg.rl.lr,
            n_steps=cfg.rl.n_steps,
            batch_size=cfg.rl.batch_size,
            ent_coef=cfg.rl.ent_coef,
            tensorboard_log=os.path.join(os.getcwd(), 'tensorboard_rl'), 
            device=device
        )
        print("Initialized new PPO agent.")

    # 6. Train the RL agent
    print("=== Training RL Module (PPO) ===")
    agent.learn(total_timesteps=cfg.rl.total_timesteps, log_interval=cfg.rl.log_freq)
    
    # Save the trained RL agent
    os.makedirs(os.path.dirname(rl_model_full_path), exist_ok=True)
    agent.save(rl_model_full_path)
    print(f"Saved PPO agent to {rl_model_full_path}")

    envs.close() 

if __name__ == '__main__':
    main()
