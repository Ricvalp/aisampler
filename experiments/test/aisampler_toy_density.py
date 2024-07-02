import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, logging
from ml_collections import config_flags

import aisampler.toy_densities as densities

from aisampler.discriminators import get_discriminator_function, plot_discriminator
from aisampler.kernels import create_henon_flow
from aisampler.kernels.utils import get_params_from_checkpoint
from aisampler.sampling import (
    metropolis_hastings_with_momentum,
    plot_chain,
    plot_samples_with_density,
    plot_kde,
)
from aisampler.sampling.metrics import effective_sample_size


_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/config.py")


def load_cfgs(
    _TASK_FILE,
):
    cfg = _TASK_FILE.value

    return cfg


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    cfg.figure_path.mkdir(parents=True, exist_ok=True)

    density = getattr(densities, cfg.target_density.name)
    density_statistics = getattr(
        densities, "statistics_" + cfg.hmc.potential_function_name
    )

    checkpoint_path = os.path.join(
        os.path.join(cfg.checkpoint_dir, cfg.target_density.name), cfg.checkpoint_name
    )

    kernel_params, discriminator_params = get_params_from_checkpoint(
        checkpoint_path=checkpoint_path,
        checkpoint_epoch=cfg.checkpoint_epoch,
        checkpoint_step=cfg.checkpoint_step,
    )

    kernel = create_henon_flow(
        num_flow_layers=cfg.kernel.num_flow_layers,
        num_hidden=cfg.kernel.num_hidden,
        num_layers=cfg.kernel.num_layers,
        d=cfg.kernel.d,
    )

    kernel_fn = jax.jit(lambda x: kernel.apply(kernel_params, x))

    samples, ar, t = metropolis_hastings_with_momentum(
        kernel_fn,
        density,
        cov_p=jnp.eye(cfg.kernel.d),
        d=cfg.kernel.d,
        parallel_chains=cfg.sample.num_parallel_chains,
        n=cfg.sample.num_iterations,
        burn_in=cfg.sample.burn_in,
        rng=jax.random.PRNGKey(cfg.seed + 2),
        vstack=False,
    )

    plot_samples_with_density(
        samples=jnp.transpose(jnp.array(samples), (1, 0, 2)),
        target_density=density,
        ar=ar,
        name=None,
    )

    plot_kde(
        samples=jnp.transpose(jnp.array(samples), (1, 0, 2)),
        name=cfg.figure_path / Path(f"kde_ring.png"),
    )

    ess = effective_sample_size(
        samples[None, :, :2],
        np.array(density_statistics["mu"]),
        np.array(density_statistics["sigma"]),
    )

    for i in range(2):
        logging.info(f"their ESS w_{i}: {ess[i]}")


if __name__ == "__main__":
    app.run(main)
