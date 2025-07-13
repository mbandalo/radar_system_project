import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def make_env(cfg, dataset, ae=None, hmm=None, gnn=None, device=None): 
    """
    Create a vectorized RadarEnv instance.
    Args:
        cfg: configuration with rl and env parameters
        dataset: instance of RadarScenesDataset
        ae: Pre-trained Autoencoder model
        hmm: Pre-trained HMM model
        gnn: Pre-trained GNN model
        device: Torch device (cuda or cpu)
    Returns:
        envs: a vectorized and optionally normalized gym environment
    """
    def _init():
        from rl.env import RadarEnv 
        env = RadarEnv(dataset=dataset, cfg=cfg, ae=ae, hmm=hmm, gnn=gnn, device=device) 
        return env

    envs = DummyVecEnv([_init for _ in range(cfg.rl.n_envs)])
    
    if getattr(cfg.rl, 'normalize_obs', False):
        envs = VecNormalize(envs, norm_obs=True, norm_reward=False)
    return envs


def make_rl_model(cfg, envs):
    """
    Instantiate and return the RL agent (PPO), optionally wrapped for meta-RL or continual learning.
    Args:
        cfg: configuration containing rl parameters
        envs: vectorized gym environments
    Returns:
        model: a Stable Baselines3 PPO model
    """
    policy = cfg.rl.policy if hasattr(cfg.rl, 'policy') else 'MlpPolicy'
    model = PPO(
        policy,
        envs,
        learning_rate=cfg.rl.lr,
        n_steps=cfg.rl.n_steps,
        batch_size=cfg.rl.batch_size,
        ent_coef=cfg.rl.ent_coef,
        verbose=1,
        tensorboard_log=getattr(cfg.rl, 'tensorboard_log', None)
    )
    return model