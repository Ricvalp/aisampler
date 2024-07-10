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

    cfg.target_density_name = "hamiltonian_mog6"
    cfg.num_steps = 40
    cfg.step_size = 0.05
    cfg.num_iterations = 1000
    cfg.burn_in = 500
    cfg.num_parallel_chains = 1

    return cfg
