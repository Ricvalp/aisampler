import jax
import jax.numpy as jnp
import numpy as np
from absl import app, logging
from ml_collections import config_flags

import aisampler.toy_densities as densities

from aisampler.sampling import hmc, plot_samples_with_density, plot_kde

from aisampler.sampling.metrics import effective_sample_size

_TASK_FILE = config_flags.DEFINE_config_file(
    "task", default="experiments/config/config_sample_hmc_toy_density.py"
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
    densities.plot_hamiltonian_density(density)

    density_statistics = getattr(
        densities, "statistics_" + cfg.target_density_name.replace("hamiltonian_", "")
    )

    grad_potential_fn = getattr(densities, f"grad_{cfg.potential_function_name}")

    samples, ar = hmc(
        density=density,
        grad_potential_fn=grad_potential_fn,
        cov_p=jnp.eye(2),
        d=2,
        parallel_chains=cfg.num_parallel_chains,
        num_steps=cfg.num_steps,
        step_size=cfg.step_size,
        n=cfg.num_iterations,
        burn_in=cfg.burn_in,
        rng=jax.random.PRNGKey(cfg.seed),
        vstack=False,
    )

    ess = effective_sample_size(
        samples[:, :, :2],
        np.array(density_statistics["mu"]),
        np.array(density_statistics["sigma"]),
    )

    for i in range(2):
        logging.info(f"ESS w_{i}: {ess[i]}")

    plot_samples_with_density(
        jnp.vstack(jnp.transpose(jnp.array(samples), (1, 0, 2))),
        target_density=density,
        ar=ar,
    )

    plot_kde(
        samples=jnp.vstack(jnp.transpose(jnp.array(samples), (1, 0, 2))),
    )


if __name__ == "__main__":
    app.run(main)
