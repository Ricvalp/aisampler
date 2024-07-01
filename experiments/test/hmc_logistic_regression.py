from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, logging
from ml_collections import config_flags

import logistic_regression
import logistic_regression.statistics as statistics
from config import load_cfgs
from logistic_regression import (
    Australian,
    German,
    Heart,
    plot_density_logistic_regression,
    plot_gradients_logistic_regression_density,
    plot_histograms2d_logistic_regression,
    plot_histograms_logistic_regression,
    plot_logistic_regression_samples,
)
from sampling import hmc
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

    grad_potential_fn = density.get_grad_energy_fn()

    samples, ar, t = hmc(
        density=density,
        grad_potential_fn=grad_potential_fn,
        cov_p=jnp.eye(density.dim) * 1.0,
        d=density.dim,
        parallel_chains=cfg.sample.num_parallel_chains,
        num_steps=cfg.hmc.num_steps,
        step_size=cfg.hmc.step_size,
        n=cfg.sample.num_iterations,
        burn_in=cfg.sample.burn_in,
        initial_std=0.1,
        rng=jax.random.PRNGKey(cfg.seed),
        )

    logging.info(f"Acceptance rate: {ar}")

    if cfg.sample.save_samples:
        cfg.sample.hmc_sample_dir.mkdir(parents=True, exist_ok=True)
        np.save(cfg.sample.hmc_sample_dir / Path(f"hmc_samples_{cfg.dataset.name}.npy"), samples)
    
    import matplotlib.pyplot as plt
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.1)
    plt.show()

    print("shape: ", samples.shape)
    print("mean: ", np.mean(samples, axis=0)[:density.dim])
    print("gt mean: ", density.mean())
    print("std: ", np.std(samples, axis=0)[density.dim:])
    print("gt std: ", density.std())

    test_density = getattr(logistic_regression, cfg.dataset.name)(
        batch_size=samples.shape[0],
        mode="test",
    )

    # v = jnp.concatenate([samples[10:100, i, :density.dim] for i in range(50)], dtype=jnp.float64)
    v = samples[:, :density.dim]
    score = np.zeros(test_density.data[0].shape[0])
    for i, (x, y) in enumerate(zip(test_density.data[0], test_density.labels[0])):
        score[i] = jax.scipy.special.logsumexp(-test_density.sigmoid(v, x, y, test_density.x_dim, test_density.y_dim)) - jnp.log(v.shape[0])

    print("average predictive posterior: ", score.mean())


if __name__ == "__main__":
    app.run(main)



        # for i in range(density.dim):

        #     eff_ess = ess(samples[:, i], density.mean()[i], density.std()[i])
        #     logging.info(f"ESS w_{i}: {eff_ess}")

        # for i in range(density.dim // 4):
        #    plot_logistic_regression_samples(
        #        samples,
        #        num_chains=cfg.sample.num_parallel_chains if cfg.sample.num_parallel_chains > 2 else None,
        #        index=i,
        #        name=cfg.figure_path / Path(f"samples_logistic_regression_{i}.png"),
        #     )
        #     plot_histograms_logistic_regression(
        #         samples,
        #         index=i,
        #         name=cfg.figure_path / Path(f"histograms_logistic_regression_{i}.png"),
        #     )
        #     plot_histograms2d_logistic_regression(
        #         samples,
        #         index=i,
        #         name=cfg.figure_path / Path(f"histograms2d_logistic_regression_{i}.png"),
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
