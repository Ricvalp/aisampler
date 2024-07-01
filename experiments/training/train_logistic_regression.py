from absl import app
from ml_collections import config_flags
from pathlib import Path

import numpy as np

import wandb
from config import load_cfgs
from trainers import TrainerLogisticRegression

from config import load_cfgs

import logistic_regression


_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/config.py")


def load_cfgs(
    _TASK_FILE,
):
    cfg = _TASK_FILE.value

    return cfg


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    cfg.figure_path.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if cfg.wandb.use:
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, config=cfg)

    density = getattr(logistic_regression, cfg.dataset.name)(
        batch_size=cfg.train.num_resampling_parallel_chains,
        mode=cfg.dataset.mode,
    )

    if cfg.train.bootstrap_with_hmc:
        hmc_samples = np.load(
            cfg.hmc_sample_dir / Path(f"hmc_samples_{cfg.dataset.name}.npy")
        )
        trainer = TrainerLogisticRegression(
            cfg=cfg,
            density=density,
            wandb_log=cfg.wandb.use,
            checkpoint_dir=cfg.checkpoint_dir,
            checkpoint_name=cfg.checkpoint_name,
            seed=cfg.seed,
            hmc_samples=hmc_samples,
        )
    else:
        trainer = TrainerLogisticRegression(
            cfg=cfg,
            density=density,
            wandb_log=cfg.wandb.use,
            checkpoint_dir=cfg.checkpoint_dir,
            checkpoint_name=cfg.checkpoint_name,
            seed=cfg.seed,
            hmc_samples=None,
        )

    trainer.train_model()


if __name__ == "__main__":
    app.run(main)
