import os
import torch
import hydra
from omegaconf import DictConfig

from rl.env import RadarEnv
from rl.agent import make_env, make_rl_model

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Train a PPO agent for radar sensor configuration using the RadarEnv with multi-label support.

    Args:
        cfg: Hydra configuration with sections:
             - data.root: directory with radar_data.h5 and scenes.json
             - calibration_params: dict for calibration
             - rl.n_envs, rl.normalize_obs, rl.policy, rl.lr, rl.n_steps, rl.batch_size, rl.ent_coef, rl.total_timesteps
    """
    # Prepare dataset and environment
    dataset = None 
    envs = make_env(cfg, dataset)

    # Instantiate RL model
    model = make_rl_model(cfg, envs)

    # Train agent
    print("=== Training PPO Agent ===")
    model.learn(total_timesteps=cfg.rl.total_timesteps)

    # Save model
    save_path = os.path.join(os.getcwd(), 'ppo_radar_model')
    model.save(save_path)
    print(f"PPO model saved to {save_path}")

if __name__ == '__main__':
    main()
