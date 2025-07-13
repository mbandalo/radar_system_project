import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import optuna
import logging
import subprocess
from torch.utils.tensorboard import SummaryWriter
from hydra.core.hydra_config import HydraConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

enable_tensorboard = True
class OptunaTensorBoardCallback:
    """Logs Optuna trial metrics and hyperparameters to TensorBoard."""
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        self.writer.add_scalar("HPO/Objective_Value", trial.value, trial.number)
        for key, value in trial.params.items():
            if isinstance(value, (int, float, bool)):
                self.writer.add_scalar(f"HPO/Hyperparameters/{key}", value, trial.number)
            else:
                self.writer.add_text(f"HPO/Hyperparameters/{key}", str(value), trial.number)
        self.writer.add_scalar("HPO/Best_Objective_Value", study.best_value, trial.number)
        self.writer.flush()


def load_scene_debug_log(trial: optuna.trial.Trial, cfg: DictConfig) -> float:
    """Parses the validation Macro F1 score from the scene module's debug log."""
    run_dir = HydraConfig.get().run.dir
    log_path = os.path.join(
        run_dir,
        f"hpo_trial_{trial.number}",
        "logs",
        "seq86",
        cfg.scene.debug_log_filename
    )

    if not os.path.isfile(log_path):
        logger.warning(f"[Trial {trial.number}] Missing debug log: {log_path}")
        return None

    try:
        with open(log_path, "r") as f:
            for line in reversed(f.readlines()):
                if "Makro F1 validacije:" in line or "Val Macro F1:" in line:
                    text = (line.split("Makro F1 validacije:")[1]
                                 if "Makro F1 validacije:" in line
                                 else line.split("Val Macro F1:")[1])
                    try:
                        score = float(text.split(",")[0].strip())
                        return score
                    except ValueError:
                        logger.error(f"[Trial {trial.number}] Failed parsing F1 from: '{line.strip()}'")
                        return None
    except Exception as e:
        logger.error(f"[Trial {trial.number}] Error reading log: {e}")
        return None

    logger.warning(f"[Trial {trial.number}] No F1 entry found in log.")
    return None


def objective(trial: optuna.Trial, cfg: DictConfig) -> float:
    """
    Objective function for Optuna HPO.
    Runs training as a subprocess and returns validation Macro F1.
    """
    from hydra.utils import get_original_cwd
    root = get_original_cwd()
    sys.path.insert(0, root)

    lr = trial.suggest_float("scene.lr", cfg.hpo.scene_hpo.lr.low, cfg.hpo.scene_hpo.lr.high, log=True)
    gnn_dropout_rate = trial.suggest_float(
        "scene.gnn_dropout_rate", cfg.hpo.scene_hpo.gnn_dropout_rate.low,
        cfg.hpo.scene_hpo.gnn_dropout_rate.high, step=cfg.hpo.scene_hpo.gnn_dropout_rate.step
    )
    gnn_weight_decay = trial.suggest_float(
        "scene.gnn_weight_decay", cfg.hpo.scene_hpo.gnn_weight_decay.low,
        cfg.hpo.scene_hpo.gnn_weight_decay.high, log=True
    )
    sampler_total_multiplier = trial.suggest_float(
        "scene.sampler_total_multiplier", cfg.hpo.scene_hpo.sampler_total_multiplier.low,
        cfg.hpo.scene_hpo.sampler_total_multiplier.high, step=cfg.hpo.scene_hpo.sampler_total_multiplier.step
    )
    target_samples_per_class = trial.suggest_int(
        "scene.target_samples_per_class", cfg.hpo.scene_hpo.target_samples_per_class.low,
        cfg.hpo.scene_hpo.target_samples_per_class.high, step=cfg.hpo.scene_hpo.target_samples_per_class.step
    )
    early_stopping_patience = trial.suggest_int(
        "scene.early_stopping_patience", cfg.hpo.scene_hpo.early_stopping_patience.low,
        cfg.hpo.scene_hpo.early_stopping_patience.high, step=cfg.hpo.scene_hpo.early_stopping_patience.step
    )

    logger.info(
        f"Trial {trial.number}: HPs: lr={lr:.6f}, dropout={gnn_dropout_rate}, weight_decay={gnn_weight_decay:.6f}, "
        f"sampler_mult={sampler_total_multiplier}, target_samples={target_samples_per_class}, "
        f"patience={early_stopping_patience}"
    )

    overrides = [
        f"scene.lr={lr}",
        f"scene.gnn_dropout_rate={gnn_dropout_rate}",
        f"scene.gnn_weight_decay={gnn_weight_decay}",
        f"scene.sampler_total_multiplier={sampler_total_multiplier}",
        f"scene.target_samples_per_class={target_samples_per_class}",
        f"scene.early_stopping_patience={early_stopping_patience}",
        f"scene.epochs={cfg.scene.epochs}",
        f"hydra.run.dir={os.path.join(HydraConfig.get().run.dir, f'hpo_trial_{trial.number}') }"
    ]

    cmd = [
        sys.executable,
        os.path.join(root, "training", "train_scene_module.py"),
        "--config-path", "../conf",
        "--config-name", "config",
        *overrides
    ]
    env = os.environ.copy()
    env['PYTHONPATH'] = root + os.pathsep + env.get('PYTHONPATH', '')

    subprocess.run(cmd, check=True, cwd=root, env=env)

    val_macro_f1 = load_scene_debug_log(trial, cfg)
    if val_macro_f1 is None:
        return -float('inf')

    logger.info(f"Trial {trial.number}: Extracted Val Macro F1 = {val_macro_f1}")
    return val_macro_f1


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    """Main function to run the Optuna HPO study."""
    study_name = "radar_segmentation_hpo_study"
    storage_name = f"sqlite:///{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True
    )

    logger.info(f"Starting HPO study for {cfg.hpo.n_trials} trials.")
    callbacks = []
    if enable_tensorboard:
        tb_log = os.path.join(os.getcwd(), "tensorboard_hpo")
        callbacks.append(OptunaTensorBoardCallback(tb_log))

    study.optimize(
        lambda trial: objective(trial, cfg),
        n_trials=cfg.hpo.n_trials,
        timeout=cfg.hpo.timeout,
        callbacks=callbacks
    )

    logger.info("\n--- HPO Study Finished ---")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info(f"Best trial: {study.best_trial.value:.4f} with params: {study.best_trial.params}")

    best_hps_path = os.path.join(os.getcwd(), "best_hpo_params.yaml")
    OmegaConf.save(config=study.best_trial.params, f=best_hps_path)
    logger.info(f"Best hyperparameters saved to {best_hps_path}")


if __name__ == "__main__":
    main()