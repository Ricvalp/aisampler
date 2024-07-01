import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, logging
from ml_collections import config_flags

import densities
import logistic_regression
from config import load_cfgs
from densities import plot_hamiltonian_density, plot_hamiltonian_density_only_q
from discriminator_models import get_discriminator_function, plot_discriminator
from kernel_models import create_henon_flow
from kernel_models.utils import get_params_from_checkpoint
from logistic_regression import (
    plot_first_kernel_iteration,
    plot_histograms2d_logistic_regression,
    plot_histograms_logistic_regression,
    plot_logistic_regression_samples,
)
from sampling import (
    metropolis_hastings_with_momentum,
    plot_chain,
    plot_samples_with_density,
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

    density = getattr(logistic_regression, cfg.dataset.name)(
        batch_size=cfg.sample.num_parallel_chains,
        mode="train",
    )

    hmc_samples = np.load(cfg.hmc_sample_dir / Path(f"hmc_samples_{cfg.dataset.name}.npy"))

    checkpoint_path = os.path.join(
        os.path.join(cfg.checkpoint_dir, cfg.dataset.name), cfg.checkpoint_name
    )

    kernel_params, discriminator_params = get_params_from_checkpoint(
        checkpoint_path=checkpoint_path,
        checkpoint_epoch=cfg.checkpoint_epoch,
        checkpoint_step=cfg.checkpoint_step,
    )

    discriminator_fn = get_discriminator_function(
        discriminator_parameters=discriminator_params,
        num_layers_psi=cfg.discriminator.num_layers_psi,
        num_hidden_psi=cfg.discriminator.num_hidden_psi,
        num_layers_eta=cfg.discriminator.num_layers_eta,
        num_hidden_eta=cfg.discriminator.num_hidden_eta,
        activation=cfg.discriminator.activation,
        d=density.dim,
    )

    kernel = create_henon_flow(
        num_flow_layers=cfg.kernel.num_flow_layers,
        num_hidden=cfg.kernel.num_hidden,
        num_layers=cfg.kernel.num_layers,
        d=density.dim,
    )

    kernel_fn = jax.jit(lambda x: kernel.apply(kernel_params, x))

    samples, ar, t = metropolis_hastings_with_momentum(
        kernel_fn,
        density,
        cov_p=jnp.eye(density.dim),
        d=density.dim,
        parallel_chains=cfg.sample.num_parallel_chains,
        n=cfg.sample.num_iterations,
        burn_in=cfg.sample.burn_in,
        rng=jax.random.PRNGKey(cfg.seed),
        starting_points=hmc_samples[:],
    )

    logging.info(f"Acceptance rate: {ar}")

    print("shape: ", samples.shape)
    print("mean: ", np.mean(samples, axis=0)[:density.dim])
    print("gt mean: ", density.mean())
    print("std: ", np.std(samples, axis=0)[density.dim:])
    print("gt std: ", density.std())

    test_density = getattr(logistic_regression, cfg.dataset.name)(
        batch_size=samples.shape[0],
        mode="test",
    )
    v = jnp.concatenate([samples[10:100, i, :test_density.dim] for i in range(50)], dtype=jnp.float64)
    score = np.zeros(test_density.data[0].shape[0])
    for i, (x, y) in enumerate(zip(test_density.data[0], test_density.labels[0])):
        score[i] = jax.scipy.special.logsumexp(-test_density.sigmoid(v, x, y, test_density.x_dim, test_density.y_dim)) - jnp.log(v.shape[0])
    print("average predictive posterior: ", score.mean())

if __name__ == "__main__":
    app.run(main)








        # plot_logistic_regression_samples(
        #     samples,
        #     num_chains=None, # cfg.sample.num_parallel_chains,
        #     index=0,
        #     name= cfg.figure_path / Path(f"samples_logistic_regression_{i}.png"),
        #     )
        # plot_histograms_logistic_regression(
        #         samples,
        #         index=0,
        #         name=cfg.figure_path / Path(f"histograms_logistic_regression_{i}.png"),
        #     )
        # plot_histograms2d_logistic_regression(
        #         samples,
        #         index=0,
        #         name=cfg.figure_path / Path(f"histograms2d_logistic_regression_{i}.png"),
        #     )
        # plot_first_kernel_iteration(
        #         kernel=kernel_fn,
        #         starting_points=hmc_samples,
        #         index=0,
        #         name=cfg.figure_path / Path(f"first_kernel_iteration_{i}.png"),
        #     )

        # their_eff_ess = effective_sample_size(
        #         samples[:, :, :density.dim],
        #         density.mean(),
        #         density.std()
        #         )
        # their_average_eff_sample_size.append(their_eff_ess)

        # for i in range(density.dim):
        #     logging.info(f"their ESS w_{i}: {their_eff_ess[i]}")

        # average_ess_per_second.append(their_eff_ess / t)

        # T += t

        # eff_ess_x = ess(samples[:, 0], density.mean()[0], density_s['sigma'][0])
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

    # their_average_eff_sample_size = np.array(their_average_eff_sample_size)
    # their_std_eff_sample_size = np.std(their_average_eff_sample_size, axis=0)
    # their_average_eff_sample_size = np.mean(their_average_eff_sample_size, axis=0)

    # average_ess_per_second = np.array(average_ess_per_second)
    # std_ess_per_second = np.std(average_ess_per_second, axis=0)
    # average_ess_per_second = np.mean(average_ess_per_second, axis=0)

    # logging.info("--------------")

    # for i in range(density.dim):
    #     logging.info(f"their Average ESS w_{i}: {their_average_eff_sample_size[i]} pm {their_std_eff_sample_size[i]}")

    # for i in range(density.dim):
    #     logging.info(f"Average ESS per second w_{i}: {average_ess_per_second[i]} pm {std_ess_per_second[i]}")
