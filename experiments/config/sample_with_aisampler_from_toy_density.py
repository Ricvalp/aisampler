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

    cfg.d = 2
    cfg.num_parallel_chains = 500
    cfg.num_iterations = 1000  # after burn-in
    cfg.burn_in = 1000

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.checkpoint_dir = pathlib.Path("./checkpoints")
    cfg.checkpoint.checkpoint_name = "debug"
    cfg.checkpoint.checkpoint_epoch = 50
    cfg.checkpoint.overwrite = True

    cfg.target_density_name = "ring"

    cfg.kernel = ConfigDict()
    cfg.kernel.num_flow_layers = 5
    cfg.kernel.num_layers = 2
    cfg.kernel.num_hidden = 32
    cfg.kernel.d = 2

    cfg.discriminator = ConfigDict()
    cfg.discriminator.num_layers_psi = 3
    cfg.discriminator.num_hidden_psi = 128
    cfg.discriminator.num_layers_eta = 3
    cfg.discriminator.num_hidden_eta = 128
    cfg.discriminator.activation = "relu"

    return cfg
