import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, logging
from ml_collections import config_flags

import aisampler.logistic_regression as logistic_regression
from aisampler.kernels import create_henon_flow, get_params_from_checkpoint, load_config
from aisampler.logistic_regression import (
    plot_histograms2d_logistic_regression,
    plot_histograms_logistic_regression,
    plot_logistic_regression_samples,
)
from aisampler.sampling import (
    metropolis_hastings_with_momentum,
)
from aisampler.sampling.metrics import effective_sample_size

_TASK_FILE = config_flags.DEFINE_config_file(
    "task", default="experiments/config/config_sample_aisampler_logistic_regression.py"
)


def load_cfgs(
    _TASK_FILE,
):
    cfg = _TASK_FILE.value

    return cfg


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    cfg.figure_path.mkdir(parents=True, exist_ok=True)

    density = getattr(logistic_regression, cfg.dataset_name)(
        batch_size=cfg.num_parallel_chains,
        mode="train",
    )

    checkpoint_path = os.path.join(cfg.checkpoint.checkpoint_dir, cfg.dataset_name)

    config = load_config(
        Path(checkpoint_path) / "cfg.json",
    )

    assert config.dataset_name == cfg.dataset_name  # sanity check

    kernel_config = config.kernel

    kernel_params, discriminator_params = get_params_from_checkpoint(
        checkpoint_path=checkpoint_path,
        checkpoint_epoch=cfg.checkpoint.checkpoint_epoch,
    )

    kernel = create_henon_flow(
        num_flow_layers=kernel_config.num_flow_layers,
        num_hidden=kernel_config.num_hidden,
        num_layers=kernel_config.num_layers,
        d=density.dim,
    )

    kernel_fn = jax.jit(lambda x: kernel.apply(kernel_params, x))

    logging.info(f"Sampling from {cfg.dataset_name} density...")

    samples, ar = metropolis_hastings_with_momentum(
        kernel_fn,
        density,
        cov_p=jnp.eye(density.dim),
        d=density.dim,
        parallel_chains=cfg.num_parallel_chains,
        n=cfg.num_iterations,
        burn_in=cfg.burn_in,
        rng=jax.random.PRNGKey(cfg.seed),
        starting_points=None,
        initial_std=0.1,
        vstack=False,
    )

    logging.info(f"Sampling done. Acceptance rate: {ar}")

    # Compute ESS

    # ess = effective_sample_size(
    #     samples[:, :, : density.dim],
    #     np.array(density.mean()),
    #     np.array(density.std()),
    # )

    # for i in range(density.dim):
    #     logging.info(f"ESS w_{i}: {ess[i]}")

    # Plot

    samples = np.vstack(np.transpose(np.array(samples), (1, 0, 2)))[:, : density.dim]

    plot_logistic_regression_samples(
        samples,
        density,
    )

    plot_histograms_logistic_regression(
        samples,
    )

    plot_histograms2d_logistic_regression(
        samples,
    )

    np.set_printoptions(linewidth=200, precision=6, suppress=True)
    print(
        "MEAN:    ",
        np.array2string(np.mean(samples, axis=0)[: density.dim], separator=","),
    )
    print("GT MEAN: ", np.array2string(density.mean(), separator=","))
    print(
        "STD:    ",
        np.array2string(np.std(samples, axis=0)[: density.dim], separator=","),
    )
    print("GT STD: ", np.array2string(density.std(), separator=","))

    test_density = getattr(logistic_regression, cfg.dataset_name)(
        batch_size=samples.shape[0],
        mode="test",
    )

    score = np.zeros(test_density.data[0].shape[0])
    for i, (x, y) in enumerate(zip(test_density.data[0], test_density.labels[0])):
        score[i] = jax.scipy.special.logsumexp(
            -test_density.sigmoid(samples, x, y, test_density.x_dim, test_density.y_dim)
        ) - jnp.log(samples.shape[0])

    logging.info(f"Average predictive posterior: {score.mean()}")


if __name__ == "__main__":
    app.run(main)
