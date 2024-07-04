from absl import app
from ml_collections import config_flags
from pathlib import Path

import jax
import numpy as np

try:
    import wandb
except ImportError:
    wandb = None


from aisampler.trainers import TrainerLogisticRegression
import aisampler.logistic_regression as logistic_regression
from aisampler.logistic_regression import (
    plot_logistic_regression_samples,
)


_TASK_FILE = config_flags.DEFINE_config_file(
    "task", default="experiments/config/config_train_logistic_regression.py"
)


def load_cfgs(
    _TASK_FILE,
):
    cfg = _TASK_FILE.value

    return cfg


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    Path(cfg.figure_path).mkdir(parents=True, exist_ok=True)
    Path(cfg.checkpoint.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if cfg.wandb.use:
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, config=cfg)

    density = getattr(logistic_regression, cfg.dataset_name)(
        batch_size=cfg.train.num_resampling_parallel_chains,
        mode="train",
    )

    trainer = TrainerLogisticRegression(
        cfg=cfg,
        density=density,
    )

    trainer.train_model()

    samples, ar = trainer.sample(
        rng=jax.random.PRNGKey(42), n=1000, burn_in=500, parallel_chains=10
    )

    plot_logistic_regression_samples(
        samples,
        density,
    )


if __name__ == "__main__":
    app.run(main)
