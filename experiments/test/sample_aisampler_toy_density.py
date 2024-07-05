import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, logging
from ml_collections import config_flags

import aisampler.toy_densities as densities

from aisampler.discriminators import get_discriminator_function, plot_discriminator
from aisampler.kernels import create_henon_flow, get_params_from_checkpoint, load_config
from aisampler.sampling import (
    get_sample_fn,
    plot_samples_with_density,
    plot_kde,
)
from aisampler.sampling.metrics import effective_sample_size


_TASK_FILE = config_flags.DEFINE_config_file(
    "task", default="experiments/config/config_sample_aisampler_toy_density.py"
)


def load_cfgs(
    _TASK_FILE,
):
    cfg = _TASK_FILE.value

    return cfg


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    cfg.figure_path.mkdir(parents=True, exist_ok=True)

    density = getattr(densities, cfg.target_density_name)

    density_statistics = getattr(
        densities, "statistics_" + cfg.target_density_name.replace("hamiltonian_", "")
    )

    checkpoint_path = os.path.join(
        cfg.checkpoint.checkpoint_dir, cfg.target_density_name
    )

    config = load_config(
        Path(checkpoint_path) / "cfg.json",
    )

    assert config.target_density_name == cfg.target_density_name  # sanity check

    kernel_config = config.kernel

    kernel_params, discriminator_params = get_params_from_checkpoint(
        checkpoint_path=checkpoint_path,
        checkpoint_epoch=cfg.checkpoint.checkpoint_epoch,
    )

    kernel = create_henon_flow(
        num_flow_layers=kernel_config.num_flow_layers,
        num_hidden=kernel_config.num_hidden,
        num_layers=kernel_config.num_layers,
        d=kernel_config.d,
    )

    kernel_fn = lambda x, params: kernel.apply(params, x)

    metropolis_hastings_with_momentum = get_sample_fn(
        kernel_fn,
        density,
        cov_p=jnp.eye(kernel_config.d),
        d=kernel_config.d,
        parallel_chains=cfg.num_parallel_chains,
        n=cfg.num_iterations,
        burn_in=cfg.burn_in,
        starting_points=None,
        vstack=False,
    )

    samples, ar = metropolis_hastings_with_momentum(
        params=kernel_params,
        rng=jax.random.PRNGKey(cfg.seed),
    )

    ess = effective_sample_size(
        samples[:, :, :2],
        np.array(density_statistics["mu"]),
        np.array(density_statistics["sigma"]),
    )

    for i in range(2):
        logging.info(f"ESS w_{i}: {ess[i]}")

    plot_samples_with_density(
        samples=jnp.vstack(jnp.transpose(jnp.array(samples), (1, 0, 2))),
        target_density=density,
        ar=ar,
    )

    plot_kde(
        samples=jnp.vstack(jnp.transpose(jnp.array(samples), (1, 0, 2))),
    )


if __name__ == "__main__":
    app.run(main)
