import pathlib
from datetime import datetime
from typing import Literal

from absl import logging
from ml_collections import ConfigDict


def get_config():

    cfg = ConfigDict()
    cfg.seed = 42

    cfg.figure_path = pathlib.Path("./figures") / datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    cfg.hmc_sample_dir = pathlib.Path("./hmc_samples")

    cfg.dataset_name = "Heart"
    cfg.d = 2
    cfg.num_parallel_chains = 10
    cfg.num_iterations = 5000  # after burn-in
    cfg.burn_in = 1000
    cfg.num_steps = 40
    cfg.step_size = 0.05

    return cfg
