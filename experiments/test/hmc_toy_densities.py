from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, logging
from ml_collections import config_flags

import aisampler.toy_densities as densities
from config import load_cfgs
from aisampler.toy_densities import plot_hamiltonian_density, plot_hamiltonian_density_only_q
from aisampler.sampling import hmc, plot_samples_with_density
from aisampler.sampling.metrics import ess, gelman_rubin_r

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
    density_statistics = getattr(densities, "statistics_" + cfg.hmc.potential_function_name)

    grad_potential_fn = getattr(densities, f"grad_{cfg.hmc.potential_function_name}")

    plot_hamiltonian_density_only_q(
        density,
        xlim_q=8, # 6.6,
        ylim_q=8, # 6.6,
        n=200,
        name=cfg.figure_path / Path(f"hamiltonian_density_{cfg.target_density.name}.png"),
    )

    samples, ar, t = hmc(
        density=density,
        grad_potential_fn=grad_potential_fn,
        cov_p=jnp.eye(cfg.sample.d),
        d=cfg.sample.d,
        parallel_chains=cfg.sample.num_parallel_chains,
        num_steps=cfg.hmc.num_steps,
        step_size=cfg.hmc.step_size,
        n=cfg.sample.num_iterations,
        burn_in=cfg.sample.burn_in,
        rng=jax.random.PRNGKey(cfg.seed),
    )

    plot_samples_with_density(
        samples,
        target_density=density,
        q_0=0.0,
        q_1=0.0,
        name=cfg.figure_path / Path(f"jumps_hmc_.png"),
        ar=None,
        c="red",
        alpha=0.6,
        linewidth=1.5,
    )

    assert True

if __name__ == "__main__":
    app.run(main)



        # logging.info(f"Acceptance rate: {ar}")
        # average_acceptance_rate.append(ar)

        # eff_ess_x = ess(samples[:, 0], density_statistics['mu'][0], density_statistics['sigma'][0])
        # logging.info(f"ESS x: {eff_ess_x}")
        # average_eff_sample_size_x.append(eff_ess_x)

        # eff_ess_y = ess(samples[:, 1], density_statistics['mu'][1], density_statistics['sigma'][1])
        # logging.info(f"ESS y: {eff_ess_y}")
        # average_eff_sample_size_y.append(eff_ess_y)



        # chains.append(samples)

    # average_eff_sample_size_x = np.array(average_eff_sample_size_x)
    # average_eff_sample_size_y = np.array(average_eff_sample_size_y)
    # average_acceptance_rate = np.array(average_acceptance_rate)

    # logging.info("------------")
    # logging.info(f"Average ESS x: {np.sum(average_eff_sample_size_x)/cfg.sample.average_results_over_trials} \pm {np.std(average_eff_sample_size_x)}")
    # logging.info(f"Average ESS y: {np.sum(average_eff_sample_size_y)/cfg.sample.average_results_over_trials} \pm {np.std(average_eff_sample_size_y)}")
    # logging.info(f"Average acceptance rate: {np.sum(average_acceptance_rate)/cfg.sample.average_results_over_trials} \pm {np.std(average_acceptance_rate)}")

    # chains = np.array(chains)[:, :, :2]
    # logging.info(f"GR R: {gelman_rubin_r(chains)}")

