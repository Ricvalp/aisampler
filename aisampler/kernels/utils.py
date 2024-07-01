import os
import orbax
import orbax.checkpoint


def get_params_from_checkpoint(checkpoint_path, checkpoint_epoch, checkpoint_step):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt = orbax_checkpointer.restore(
        os.path.join(checkpoint_path, f"{checkpoint_epoch}_{checkpoint_step}")
    )
    L_state = ckpt["L"]
    D_state = ckpt["D"]
    return {"params": L_state["params"]}, {"params": {"D": D_state["params"]}}
