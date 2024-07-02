import pathlib
from datetime import datetime
from typing import Literal

from absl import logging
from ml_collections import ConfigDict


def get_config(mode: Literal["train", "sample"] = None):
    if mode is None:
        mode = "train"
        logging.info(f"No mode provided, using '{mode}' as default")

    cfg = ConfigDict()
    cfg.seed = 42

    cfg.figure_path = pathlib.Path("./figures") / datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.checkpoint_dir = pathlib.Path("./checkpoints")
    cfg.checkpoint.checkpoint_name = "debug"
    cfg.checkpoint.overwrite = True

    # Target density
    cfg.target_density = ConfigDict()
    cfg.target_density.name = "ring"

    # Wandb
    cfg.wandb = ConfigDict()
    cfg.wandb.use = False
    cfg.wandb.project = "adversarial-involutive-sampler-debug"
    cfg.wandb.entity = "ricvalp"

    # Kernel
    cfg.kernel = ConfigDict()
    cfg.kernel.num_flow_layers = 5
    cfg.kernel.num_layers = 2
    cfg.kernel.num_hidden = 32
    cfg.kernel.d = 2

    # Discriminator
    cfg.discriminator = ConfigDict()
    cfg.discriminator.num_layers_psi = 3
    cfg.discriminator.num_hidden_psi = 128
    cfg.discriminator.num_layers_eta = 3
    cfg.discriminator.num_hidden_eta = 128
    cfg.discriminator.activation = "relu"

    # Train
    cfg.train = ConfigDict()
    cfg.train.init = "glorot_normal"
    cfg.train.kernel_learning_rate = 1e-4
    cfg.train.discriminator_learning_rate = 1e-4
    cfg.train.num_resampling_steps = 100
    cfg.train.num_resampling_parallel_chains = 500
    cfg.train.resampling_burn_in = 0
    cfg.train.batch_size = 4096
    cfg.train.num_epochs = 50
    cfg.train.num_AR_steps = 1
    cfg.train.num_adversarial_steps = 1

    # Log
    cfg.log = ConfigDict()
    cfg.log.save_every = 50

    # cfg.log.num_steps = 1000
    # cfg.log.num_parallel_chains = 2
    # cfg.log.burn_in = 100
    # cfg.log.samples_to_plot = 5000

    if mode == "sample":
        cfg.sample = ConfigDict()
        cfg.sample.d = 2
        cfg.sample.num_parallel_chains = 500
        cfg.sample.num_iterations = 1000  # after burn-in
        cfg.sample.burn_in = 1000

        cfg.sample.average_results_over_trials = 5
        cfg.sample.save_samples = False
        cfg.sample.hmc_sample_dir = pathlib.Path("./hmc_samples")

        cfg.hmc = ConfigDict()
        cfg.hmc.potential_function_name = "mog6"
        cfg.hmc.num_steps = 40
        cfg.hmc.step_size = 0.05

    return cfg
