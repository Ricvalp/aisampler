import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, logging
from ml_collections import config_flags

import densities
from config import load_cfgs
from densities import plot_hamiltonian_density, plot_hamiltonian_density_only_q
from discriminator_models import get_discriminator_function, plot_discriminator
from kernel_models import create_henon_flow
from kernel_models.utils import get_params_from_checkpoint
from sampling import (
    metropolis_hastings_with_momentum,
    plot_chain,
    plot_samples_with_density,
    plot_kde,
)
from sampling.metrics import effective_sample_size, ess, gelman_rubin_r


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

    # discriminator_fn = get_discriminator_function(
    #     discriminator_parameters=discriminator_params,
    #     num_layers_psi=cfg.discriminator.num_layers_psi,
    #     num_hidden_psi=cfg.discriminator.num_hidden_psi,
    #     num_layers_eta=cfg.discriminator.num_layers_eta,
    #     num_hidden_eta=cfg.discriminator.num_hidden_eta,
    #     activation=cfg.discriminator.activation,
    #     d=cfg.kernel.d,
    # )

    # x_0s = [
    #     5 * jnp.array([jnp.sin((i * jnp.pi) / 3), jnp.cos((i * jnp.pi) / 3)]) for i in range(6)
    # ] + [jnp.array([0.0, 0.0])]
    # i = 0
    # for x_0 in x_0s:
    #     i += 1
    #     plot_discriminator(
    #         discriminator_fn,
    #         xlim_q=6.5,
    #         ylim_q=6.5,
    #         xlim_p=3,
    #         ylim_p=3,
    #         n=100,
    #         x_0=x_0,  # jnp.array([.0, .0]),
    #         p_0=0.0,
    #         p_1=0.0,
    #         name=cfg.figure_path / Path(f"discriminator_{i}"),
    #     )

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
    )

    # plot_samples_with_density(
    #     samples,
    #     target_density=density,
    #     q_0=0.0,
    #     q_1=0.0,
    #     name=cfg.figure_path / Path(f"jumps_mog6.png"),
    #     ar=None,
    #     c="red",
    #     alpha=0.6,
    #     linewidth=1.5,
    # )

    plot_kde(samples, name=cfg.figure_path / Path(f"kde_ring.png"))

    assert True

    # plot_chain(
    #     samples[:100],
    #     target_density=density,
    #     q_0=0.0,
    #     q_1=0.0,
    #     name=cfg.figure_path / Path(f"samples_{i}.png"),
    #     ar=ar,
    #     c="red",
    #     linewidth=0.5,
    # )

    #     chains.append(samples)

    # average_eff_sample_size_x = np.array(average_eff_sample_size_x)
    # average_eff_sample_size_y = np.array(average_eff_sample_size_y)
    # average_acceptance_rate = np.array(average_acceptance_rate)

    # logging.info("------------")
    # logging.info(f"Average ESS x: {np.sum(average_eff_sample_size_x)/cfg.sample.average_results_over_trials} \pm {np.std(average_eff_sample_size_x)}")
    # logging.info(f"Average ESS y: {np.sum(average_eff_sample_size_y)/cfg.sample.average_results_over_trials} \pm {np.std(average_eff_sample_size_y)}")
    # logging.info(f"Average acceptance rate: {np.sum(average_acceptance_rate)/cfg.sample.average_results_over_trials} \pm {np.std(average_acceptance_rate)}")

    # logging.info("------------")

    # their_average_eff_sample_size = np.array(their_average_eff_sample_size)
    # their_std_eff_sample_size = np.std(their_average_eff_sample_size, axis=0)
    # their_average_eff_sample_size = np.mean(their_average_eff_sample_size, axis=0)

    # average_ess_per_second = np.array(average_ess_per_second)
    # std_ess_per_second = np.std(average_ess_per_second, axis=0)
    # average_ess_per_second = np.mean(average_ess_per_second, axis=0)

    # logging.info("--------------")

    # for i in range(2):
    #     logging.info(f"their Average ESS w_{i}: {their_average_eff_sample_size[i]} pm {their_std_eff_sample_size[i]}")

    # for i in range(2):
    #     logging.info(f"Average ESS per second w_{i}: {average_ess_per_second[i]} pm {std_ess_per_second[i]}")

    # chains = np.array(chains)[:, :, :2]
    # logging.info(f"GR R: {gelman_rubin_r(chains)}")

    # logging.info(f"Acceptance rate: {ar}")
    # average_acceptance_rate.append(ar)

    # eff_ess_x = ess(samples[:, 0], density_statistics['mu'][0], density_statistics['sigma'][0])
    # logging.info(f"ESS x: {eff_ess_x}")
    # average_eff_sample_size_x.append(eff_ess_x)

    # eff_ess_y = ess(samples[:, 1], density_statistics['mu'][1], density_statistics['sigma'][1])
    # logging.info(f"ESS y: {eff_ess_y}")
    # average_eff_sample_size_y.append(eff_ess_y)

    # their_eff_ess = effective_sample_size(
    #         samples[None, :, :2],
    #         np.array(density_statistics['mu']),
    #         np.array(density_statistics['sigma'])
    #         )

    # their_average_eff_sample_size.append(their_eff_ess)
    # for i in range(2):
    #     logging.info(f"their ESS w_{i}: {their_eff_ess[i]}")

    # average_ess_per_second.append(their_eff_ess / t)


if __name__ == "__main__":
    app.run(main)
