import pathlib
from datetime import datetime
from ml_collections import ConfigDict


def get_config():

    cfg = ConfigDict()
    cfg.seed = 42

    cfg.figure_path = pathlib.Path("./figures") / datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    cfg.dataset_name = "Australian"

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.checkpoint_dir = "./checkpoints"
    cfg.checkpoint.checkpoint_epoch = 50
    cfg.checkpoint.overwrite = True

    cfg.d = 2
    cfg.num_parallel_chains = 500
    cfg.num_iterations = 1000  # after burn-in
    cfg.burn_in = 1000

    return cfg
