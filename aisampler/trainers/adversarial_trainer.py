import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
from absl import logging
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

import wandb
from aisampler.discriminators import create_simple_discriminator, log_plot
from aisampler.sampling import (
    metropolis_hastings_with_momentum,
)

from aisampler.sampling import hmc


from aisampler.trainers.utils import SamplesDataset, numpy_collate
from aisampler.sampling.metrics import ess

from aisampler.logistic_regression import (
    plot_logistic_regression_samples,
    plot_histograms_logistic_regression,
    plot_histograms2d_logistic_regression,
    plot_first_kernel_iteration,
)


class Trainer:
    def __init__(
        self,
        cfg,
        density,
    ):
        self.rng = jax.random.PRNGKey(cfg.seed)

        self.density = density
        self.wandb_log = cfg.wandb.use
        self.checkpoint_path = os.path.join(
            cfg.checkpoint.checkpoint_dir, cfg.target_density_name
        )

        self.cfg = cfg

        self.init_model()
        self.create_train_steps()

    def init_model(self):

        discriminator = create_simple_discriminator(
            num_flow_layers=self.cfg.kernel.num_flow_layers,
            num_hidden_flow=self.cfg.kernel.num_hidden,
            num_layers_flow=self.cfg.kernel.num_layers,
            num_layers_psi=self.cfg.discriminator.num_layers_psi,
            num_hidden_psi=self.cfg.discriminator.num_hidden_psi,
            num_layers_eta=self.cfg.discriminator.num_layers_eta,
            num_hidden_eta=self.cfg.discriminator.num_hidden_eta,
            activation=self.cfg.discriminator.activation,
            d=self.cfg.kernel.d,
        )

        self.rng, init_rng, init_points_rng = jax.random.split(self.rng, 3)

        discriminator_params = discriminator.init(
            init_rng, jax.random.normal(init_points_rng, (100, 2 * self.cfg.kernel.d))
        )["params"]

        theta_params = discriminator_params["L"]
        phi_params = discriminator_params["D"]

        L_optimizer = optax.adam(learning_rate=self.cfg.train.kernel_learning_rate)
        discriminator_optimizer = optax.adam(
            learning_rate=self.cfg.train.discriminator_learning_rate
        )
        self.L_state = TrainState.create(
            apply_fn=discriminator.L.apply, params=theta_params, tx=L_optimizer
        )
        self.D_state = TrainState.create(
            apply_fn=discriminator.apply, params=phi_params, tx=discriminator_optimizer
        )

    def create_train_steps(self):
        def maximize_AR_step(L_state, D_state, batch):
            loss_fn = lambda theta_params: AR_loss(theta_params, D_state, batch)
            ar_loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(L_state.params)
            L_state = L_state.apply_gradients(grads=grads)

            return L_state, ar_loss

        self.maximize_AR_step = jax.jit(maximize_AR_step)

        def minimize_adversarial_loss_step(L_state, D_state, batch):
            my_loss = lambda phi_params: adversarial_loss(
                phi_params, D_state, L_state, batch
            )
            adv_loss, grads = jax.value_and_grad(my_loss, has_aux=False)(D_state.params)
            D_state = D_state.apply_gradients(grads=grads)

            return D_state, adv_loss

        self.minimize_adversarial_loss_step = jax.jit(minimize_adversarial_loss_step)

    def create_data_loader(self, key):
        key, subkey = jax.random.split(key)

        samples, ar = self.sample(
            rng=subkey,
            n=self.cfg.train.num_resampling_steps,
            burn_in=self.cfg.train.resampling_burn_in,
            parallel_chains=self.cfg.train.num_resampling_parallel_chains,
        )

        dataset = SamplesDataset(np.array(samples))
        self.data_loader = DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            collate_fn=numpy_collate,
        )

        return key, ar

    def train_epoch(self, epoch_idx):
        self.rng, ar = self.create_data_loader(self.rng)
        if self.wandb_log:
            wandb.log({"acceptance rate": ar})

        ar_losses = []
        adv_losses = []
        for i, batch in enumerate(self.data_loader):
            for _ in range(self.cfg.train.num_AR_steps):
                self.L_state, ar_loss = self.maximize_AR_step(
                    self.L_state, self.D_state, batch
                )
            ar_losses.append(ar_loss)

            for _ in range(self.cfg.train.num_adversarial_steps):
                self.D_state, adv_loss = self.minimize_adversarial_loss_step(
                    self.L_state, self.D_state, batch
                )
            adv_losses.append(adv_loss)

            if self.wandb_log:
                wandb.log(
                    {
                        "Epoch": epoch_idx,
                        "AR loss": ar_loss,
                        "adversarial loss": adv_loss,
                    }
                )

        if epoch_idx % self.cfg.checkpoint.save_every == 0:
            self.save_model(epoch=epoch_idx)

        return jnp.array(ar_losses).mean(), jnp.array(adv_losses).mean(), ar

    def train_model(self):
        for epoch in tqdm(range(self.cfg.train.num_epochs)):
            ar_loss, adv_loss, ar = self.train_epoch(epoch_idx=epoch)
            tqdm.write(
                f"Epoch {epoch}: ar_loss={ar_loss:.4f}, adv_loss={adv_loss:.4f}, ar={ar:.4f}"
            )

    def sample(self, rng, n, burn_in, parallel_chains):

        kernel_fn = jax.jit(
            lambda x: self.L_state.apply_fn({"params": self.L_state.params}, x)
        )

        samples, ar = metropolis_hastings_with_momentum(
            kernel=kernel_fn,
            density=self.density,
            d=self.cfg.kernel.d,
            n=n,
            cov_p=jnp.eye(self.cfg.kernel.d),
            parallel_chains=parallel_chains,
            burn_in=burn_in,
            rng=rng,
        )

        return samples, ar

    def save_model(self, epoch):
        ckpt = {"L": self.L_state, "D": self.D_state}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        orbax_checkpointer.save(
            (Path(self.checkpoint_path) / f"{epoch}").absolute(),
            ckpt,
            save_args=save_args,
            force=self.cfg.checkpoint.overwrite,
        )

        if epoch == 0:
            with open(os.path.join(self.checkpoint_path, "cfg.json"), "w") as f:
                json.dump(self.cfg.to_dict(), f, indent=4)


class TrainerLogisticRegression:
    def __init__(
        self,
        cfg,
        density,
    ):
        self.rng = jax.random.PRNGKey(cfg.seed)

        self.density = density
        self.wandb_log = cfg.wandb.use
        self.checkpoint_path = os.path.join(
            cfg.checkpoint.checkpoint_dir, cfg.dataset_name
        )

        self.cfg = cfg

        self.hmc_samples = None

        self.init_model()
        self.create_train_steps()

    def init_model(self):
        discriminator = create_simple_discriminator(
            num_flow_layers=self.cfg.kernel.num_flow_layers,
            num_hidden_flow=self.cfg.kernel.num_hidden,
            num_layers_flow=self.cfg.kernel.num_layers,
            num_layers_psi=self.cfg.discriminator.num_layers_psi,
            num_hidden_psi=self.cfg.discriminator.num_hidden_psi,
            num_layers_eta=self.cfg.discriminator.num_layers_eta,
            num_hidden_eta=self.cfg.discriminator.num_hidden_eta,
            activation=self.cfg.discriminator.activation,
            d=self.density.dim,
        )

        self.rng, init_rng, init_points_rng = jax.random.split(self.rng, 3)

        discriminator_params = discriminator.init(
            init_rng, jax.random.normal(init_points_rng, (100, 2 * self.density.dim))
        )["params"]

        theta_params = discriminator_params["L"]
        phi_params = discriminator_params["D"]

        L_optimizer = optax.adam(learning_rate=self.cfg.train.kernel_learning_rate)
        discriminator_optimizer = optax.adam(
            learning_rate=self.cfg.train.discriminator_learning_rate
        )
        self.L_state = TrainState.create(
            apply_fn=discriminator.L.apply, params=theta_params, tx=L_optimizer
        )
        self.D_state = TrainState.create(
            apply_fn=discriminator.apply, params=phi_params, tx=discriminator_optimizer
        )

    def create_train_steps(self):
        def maximize_AR_step(L_state, D_state, batch):
            loss_fn = lambda theta_params: AR_loss(theta_params, D_state, batch)
            ar_loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(L_state.params)
            L_state = L_state.apply_gradients(grads=grads)

            return L_state, ar_loss

        self.maximize_AR_step = jax.jit(maximize_AR_step)

        def minimize_adversarial_loss_step(L_state, D_state, batch):
            my_loss = lambda phi_params: adversarial_loss(
                phi_params, D_state, L_state, batch
            )
            adv_loss, grads = jax.value_and_grad(my_loss, has_aux=False)(D_state.params)
            D_state = D_state.apply_gradients(grads=grads)

            return D_state, adv_loss

        self.minimize_adversarial_loss_step = jax.jit(minimize_adversarial_loss_step)

    def create_data_loader(self, key):

        key, subkey = jax.random.split(key)
        samples, ar = self.sample(
            rng=subkey,
            n=self.cfg.train.num_resampling_steps,
            burn_in=self.cfg.train.resampling_burn_in,
            parallel_chains=self.cfg.train.num_resampling_parallel_chains,
        )

        dataset = SamplesDataset(np.array(samples))
        self.data_loader = DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            collate_fn=numpy_collate,
        )

        return key, ar

    def create_data_loader_with_hmc(self, key):

        Path(f"./data/hmc_samples").mkdir(exist_ok=True)
        hmc_samples_file = Path(f"./data/hmc_samples/{self.cfg.dataset_name}.npy")

        if os.path.exists(hmc_samples_file):
            logging.info(
                f"Loading HMC samples from {hmc_samples_file} for bootstrapping."
            )
            hmc_samples = np.load(hmc_samples_file)
        else:
            logging.info("Sampling with HMC for bootstrapping.")
            hmc_samples, ar = sample_with_hmc(
                density=self.density, hmc_bootstrapping_cfg=self.cfg.hmc_bootstrapping
            )
            logging.info(f"Sampling done. Acceptance rate: {ar}")
            np.save(hmc_samples_file, hmc_samples)
            logging.info(f"Saved HMC samples to {hmc_samples_file}.")

        self.hmc_samples = hmc_samples

        dataset = SamplesDataset(np.array(hmc_samples))
        self.data_loader = DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            collate_fn=numpy_collate,
        )

        return key, -1.0

    def train_epoch(self, epoch_idx):
        if self.cfg.hmc_bootstrapping.use and epoch_idx == 0:
            self.rng, ar = self.create_data_loader_with_hmc(self.rng)
        elif (
            not self.cfg.hmc_bootstrapping.use
            or epoch_idx > self.cfg.hmc_bootstrapping.num_epochs
        ):
            self.rng, ar = self.create_data_loader(self.rng)
            if self.wandb_log:
                wandb.log({"acceptance rate": ar})
        else:
            ar = -1.0

        ar_losses = []
        adv_losses = []
        for i, batch in enumerate(self.data_loader):
            for _ in range(self.cfg.train.num_AR_steps):
                self.L_state, ar_loss = self.maximize_AR_step(
                    self.L_state, self.D_state, batch
                )
            ar_losses.append(ar_loss)
            for _ in range(self.cfg.train.num_adversarial_steps):
                self.D_state, adv_loss = self.minimize_adversarial_loss_step(
                    self.L_state, self.D_state, batch
                )
            adv_losses.append(adv_loss)

            if self.wandb_log:
                wandb.log(
                    {
                        "Epoch": epoch_idx,
                        "AR loss": ar_loss,
                        "adversarial loss": adv_loss,
                    }
                )

        return jnp.array(ar_losses).mean(), jnp.array(adv_losses).mean(), ar

    def train_model(self):

        self.train_epoch(epoch_idx=0)  # for logging purposes

        for epoch in tqdm(
            range(1, self.cfg.hmc_bootstrapping.num_epochs + self.cfg.train.num_epochs)
        ):
            rng, subkey = jax.random.split(self.rng)
            ar_loss, adv_loss, ar = self.train_epoch(epoch_idx=epoch)
            tqdm.write(
                f"Epoch {epoch}: ar_loss={ar_loss:.4f}, adv_loss={adv_loss:.4f}, ar={ar:.4f}"
            )
            if epoch % self.cfg.checkpoint.save_every == 0:
                self.save_model(epoch=epoch)
                _, ar = self.sample(
                    rng=subkey,
                    n=self.cfg.train.num_resampling_steps,
                    burn_in=self.cfg.train.resampling_burn_in,
                    parallel_chains=self.cfg.train.num_resampling_parallel_chains,
                )

    def sample(self, rng, n, burn_in, parallel_chains):
        kernel_fn = jax.jit(
            lambda x: self.L_state.apply_fn({"params": self.L_state.params}, x)
        )
        samples, ar = metropolis_hastings_with_momentum(
            kernel=kernel_fn,
            density=self.density,
            d=self.density.dim,
            n=n,
            cov_p=jnp.eye(self.density.dim),
            parallel_chains=parallel_chains,
            burn_in=burn_in,
            rng=rng,
            initial_std=1.0,
            starting_points=self.hmc_samples,
        )

        if self.wandb_log:

            index = 0

            fig = self.cfg.figure_path / str(np.random.randint(999999))
            plot_logistic_regression_samples(
                samples[: self.cfg.log.samples_to_plot],
                num_chains=None,
                index=index,
                name=fig,  # Path(f"samples_logistic_regression_{index}.png"),
            )
            wandb.log(
                {
                    f"samples with {parallel_chains} chains": wandb.Image(
                        str(fig) + ".png"
                    )
                }
            )
            os.remove(str(fig) + ".png")

            fig1 = self.cfg.figure_path / str(np.random.randint(999999))
            plot_histograms_logistic_regression(
                samples[: self.cfg.log.samples_to_plot],
                index=index,
                name=fig1,  # Path(f"histograms_logistic_regression_{index}.png"),
            )
            wandb.log(
                {
                    f"histograms with {parallel_chains} chains": wandb.Image(
                        str(fig1) + ".png"
                    )
                }
            )
            os.remove(str(fig1) + ".png")

            fig2 = self.cfg.figure_path / str(np.random.randint(999999))
            plot_histograms2d_logistic_regression(
                samples[: self.cfg.log.samples_to_plot],
                index=index,
                name=fig2,  # Path(f"histograms2d_logistic_regression_{index}.png"),
            )
            wandb.log(
                {
                    f"histograms2d with {parallel_chains} chains": wandb.Image(
                        str(fig2) + ".png"
                    )
                }
            )
            os.remove(str(fig2) + ".png")

            fig3 = self.cfg.figure_path / str(np.random.randint(999999))
            plot_first_kernel_iteration(
                kernel=kernel_fn,
                starting_points=self.hmc_samples,
                index=index,
                name=fig3,  # Path(f"first_kernel_iteration_{index}.png"),
            )
            wandb.log(
                {
                    f"first_kernel_iteration with {parallel_chains} chains": wandb.Image(
                        str(fig3) + ".png"
                    )
                }
            )
            os.remove(str(fig3) + ".png")

            fig3p = self.cfg.figure_path / str(np.random.randint(999999))
            plot_first_kernel_iteration(
                kernel=kernel_fn,
                starting_points=self.hmc_samples,
                index=18,
                name=fig3p,  # Path(f"first_kernel_iteration_{index}.png"),
            )
            wandb.log(
                {
                    f"first_kernel_iteration (momenta) with {parallel_chains} chains": wandb.Image(
                        str(fig3p) + ".png"
                    )
                }
            )
            os.remove(str(fig3p) + ".png")

            wandb.log({"acceptance rate": ar})

            esss = []
            for i in range(self.density.dim):
                eff_ess = ess(
                    samples[:1000, i], self.density.mean()[i], self.density.std()[i]
                )
                esss.append(eff_ess)
                # wandb.log({f"ESS w_{i}": eff_ess})
            wandb.log({f"Minimum ESS (1000 max)": np.min(esss)})
            wandb.log({f"Average ESS (1000 max)": np.mean(esss)})

        return samples, ar

    def save_model(self, epoch):
        ckpt = {"L": self.L_state, "D": self.D_state}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        orbax_checkpointer.save(
            (Path(self.checkpoint_path) / f"{epoch}").absolute(),
            ckpt,
            save_args=save_args,
            force=self.cfg.checkpoint.overwrite,
        )

        if epoch == 0:
            with open(os.path.join(self.checkpoint_path, "cfg.json"), "w") as f:
                json.dump(self.cfg.to_dict(), f, indent=4)


def r(y):
    return 1 / (1 + jnp.exp(-y))


def AR_loss(theta_params, D_state, batch):
    Dx = D_state.apply_fn(
        {
            "params": {
                "L": theta_params,
                "D": D_state.params,
            }
        },
        batch,
    )

    return -(r(Dx)).mean()


def adversarial_loss(phi_params, D_state, L_state, batch):
    Dx = D_state.apply_fn(
        {
            "params": {
                "L": L_state.params,
                "D": phi_params,
            }
        },
        batch,
    )

    return (r(Dx) * jnp.log(r(Dx))).mean()


def sample_with_hmc(density, hmc_bootstrapping_cfg):

    hmc_density = density.new_instance(
        new_batch_size=hmc_bootstrapping_cfg.num_parallel_chains,
    )

    samples, ar = hmc(
        density=hmc_density,
        grad_potential_fn=density.get_grad_energy_fn(),
        cov_p=jnp.eye(density.dim) * 1.0,
        d=density.dim,
        parallel_chains=hmc_bootstrapping_cfg.num_parallel_chains,
        num_steps=hmc_bootstrapping_cfg.num_steps,
        step_size=hmc_bootstrapping_cfg.step_size,
        n=hmc_bootstrapping_cfg.num_iterations,
        burn_in=hmc_bootstrapping_cfg.burn_in,
        initial_std=hmc_bootstrapping_cfg.initial_std,
        rng=jax.random.PRNGKey(hmc_bootstrapping_cfg.seed),
    )

    return samples, ar
