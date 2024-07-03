import os
import orbax
import orbax.checkpoint
from pathlib import Path
from typing import Union
import json
from ml_collections import ConfigDict


def get_params_from_checkpoint(
    checkpoint_path: Union[str, Path], checkpoint_epoch: int
) -> dict:
    """
    Args:
        checkpoint_path: str or Path
        checkpoint_epoch: int
    Returns:
        tuple: Tuple of dictionaries containing the parameters of the kernel and the discriminator.
    """
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt = orbax_checkpointer.restore(
        os.path.join(checkpoint_path, f"{checkpoint_epoch}")
    )
    L_state = ckpt["L"]
    D_state = ckpt["D"]
    return {"params": L_state["params"]}, {"params": {"D": D_state["params"]}}


def load_config(filename):
    with open(filename, "r") as f:
        cfg = json.load(f)
    return ConfigDict(cfg)
