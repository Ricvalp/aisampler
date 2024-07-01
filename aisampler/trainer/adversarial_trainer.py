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

import wandb
from discriminators import create_simple_discriminator, log_plot
from sampling import metropolis_hastings_with_momentum, plot_samples_with_density
from trainer.utils import SamplesDataset, numpy_collate
from sampling.metrics import ess, gelman_rubin_r

from logistic_regression import (
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
        wandb_log,
        checkpoint_dir,
        checkpoint_name,
        seed,
    ):
        self.rng = jax.random.PRNGKey(seed)

        self.density = density
        self.wandb_log = wandb_log
        self.checkpoint_path = os.path.join(
            os.path.join(checkpoint_dir, cfg.target_density.name), checkpoint_name
        )

        self.cfg = cfg

        self.init_model()
        self.create_train_steps()

        # save cfg into checkpoint_path

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

    def create_data_loader(self, key, epoch_idx):
        key, subkey = jax.random.split(key)

        samples, _ = self.sample(
            rng=subkey,
            n=self.cfg.train.num_resampling_steps,
            burn_in=self.cfg.train.resampling_burn_in,
            parallel_chains=self.cfg.train.num_resampling_parallel_chains,
            name=None,  # f"samples_in_data_loader_epoch_{epoch_idx}.png",
        )

        dataset = SamplesDataset(np.array(samples))
        self.data_loader = DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            collate_fn=numpy_collate,
        )

        return key

    def train_epoch(self, epoch_idx):
        self.rng = self.create_data_loader(self.rng, epoch_idx)
        for i, batch in enumerate(self.data_loader):
            for _ in range(self.cfg.train.num_AR_steps):
                self.L_state, ar_loss = self.maximize_AR_step(
                    self.L_state, self.D_state, batch
                )
            for _ in range(self.cfg.train.num_adversarial_steps):
                self.D_state, adv_loss = self.minimize_adversarial_loss_step(
                    self.L_state, self.D_state, batch
                )

            print(
                f"Epoch: {epoch_idx}, AR loss: {ar_loss}, adversarial loss: {adv_loss}"
            )

            if self.wandb_log:
                wandb.log(
                    {
                        "Epoch": epoch_idx,
                        "AR loss": ar_loss,
                        "adversarial loss": adv_loss,
                    }
                )

            if i % self.cfg.log.log_every == 0:
                _, ar = self.sample(
                    rng=self.rng,
                    n=self.cfg.log.num_steps,
                    burn_in=self.cfg.log.burn_in,
                    parallel_chains=self.cfg.log.num_parallel_chains,
                    name=None,  # f"samples_{epoch_idx}.png",
                )
                self.save_model(epoch=epoch_idx, step=i)

                if self.wandb_log:
                    wandb.log({"acceptance rate": ar})

                    fig = log_plot(
                        discriminator_parameters={"params": {"D": self.D_state.params}},
                        num_layers_psi=self.cfg.discriminator.num_layers_psi,
                        num_hidden_psi=self.cfg.discriminator.num_hidden_psi,
                        num_layers_eta=self.cfg.discriminator.num_layers_eta,
                        num_hidden_eta=self.cfg.discriminator.num_hidden_eta,
                        activation=self.cfg.discriminator.activation,
                        d=self.cfg.kernel.d,
                        name="discriminator",
                    )
                    wandb.log({"discriminator": fig})

    def train_model(self):
        for epoch in range(self.cfg.train.num_epochs):
            self.train_epoch(epoch_idx=epoch)

    def sample(self, rng, n, burn_in, parallel_chains, name):
        kernel_fn = jax.jit(
            lambda x: self.L_state.apply_fn({"params": self.L_state.params}, x)
        )
        logging.info("Sampling...")
        start_time = time.time()
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
        logging.info(f"Sampling took {time.time() - start_time} seconds")

        if name is not None:
            name = self.cfg.figure_path / Path(name)

        fig = plot_samples_with_density(
            samples,
            target_density=self.density,
            q_0=0.0,
            q_1=0.0,
            name=name,
            s=0.5,
            c="red",
            alpha=0.05,
            ar=ar,
        )

        if self.wandb_log:
            wandb.log({f"samples with {parallel_chains} chains": fig})

        return samples, ar

    def save_model(self, epoch, step):
        ckpt = {"L": self.L_state, "D": self.D_state}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        orbax_checkpointer.save(
            os.path.join(self.checkpoint_path, f"{epoch}_{step}"),
            ckpt,
            save_args=save_args,
        )

        # log cfg into checkpoint_path
        if epoch == 0 and step == 0:
            with open(os.path.join(self.checkpoint_path, "cfg.txt"), "w") as f:
                f.write(str(self.cfg))

    def load_model(self, epoch, step):
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        ckpt = orbax_checkpointer.restore(
            os.path.join(self.checkpoint_path, f"{epoch}_{step}")
        )
        self.L_state = ckpt["L"]
        self.D_state = ckpt["D"]


class TrainerLogisticRegression:
    def __init__(
        self,
        cfg,
        density,
        wandb_log,
        checkpoint_dir,
        checkpoint_name,
        seed,
        hmc_samples=None,
    ):
        self.rng = jax.random.PRNGKey(seed)

        self.density = density
        self.wandb_log = wandb_log
        self.checkpoint_path = os.path.join(
            os.path.join(checkpoint_dir, cfg.dataset.name), checkpoint_name
        )

        self.cfg = cfg

        self.hmc_samples = hmc_samples

        self.init_model()
        self.create_train_steps()

        # save cfg into checkpoint_path

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

    def create_data_loader(self, key, epoch_idx, hmc_samples=False):

        if hmc_samples:
            dataset = SamplesDataset(np.array(self.hmc_samples))
            self.data_loader = DataLoader(
                dataset,
                batch_size=self.cfg.train.batch_size,
                shuffle=True,
                collate_fn=numpy_collate,
            )
            return key

        else:
            key, subkey = jax.random.split(key)
            samples, ar = self.sample(
                rng=subkey,
                n=self.cfg.train.num_resampling_steps,
                burn_in=self.cfg.train.resampling_burn_in,
                parallel_chains=self.cfg.train.num_resampling_parallel_chains,
                name=None,  # f"samples_in_data_loader_epoch_{epoch_idx}.png",
            )

            dataset = SamplesDataset(np.array(samples))
            self.data_loader = DataLoader(
                dataset,
                batch_size=self.cfg.train.batch_size,
                shuffle=True,
                collate_fn=numpy_collate,
            )

            print(f"ACCEPTANCE RATE: {ar}")
            return key

    def train_epoch(self, epoch_idx):
        if self.hmc_samples is not None:
            self.rng = self.create_data_loader(self.rng, epoch_idx, hmc_samples=True)
        else:
            self.rng = self.create_data_loader(self.rng, epoch_idx, hmc_samples=False)

        for i, batch in enumerate(self.data_loader):
            for _ in range(self.cfg.train.num_AR_steps):
                self.L_state, ar_loss = self.maximize_AR_step(
                    self.L_state, self.D_state, batch
                )
            for _ in range(self.cfg.train.num_adversarial_steps):
                self.D_state, adv_loss = self.minimize_adversarial_loss_step(
                    self.L_state, self.D_state, batch
                )

            print(
                f"Epoch: {epoch_idx}, AR loss: {ar_loss}, adversarial loss: {adv_loss}"
            )

            if self.wandb_log:
                wandb.log(
                    {
                        "Epoch": epoch_idx,
                        "AR loss": ar_loss,
                        "adversarial loss": adv_loss,
                    }
                )

    def train_model(self):
        if self.hmc_samples is not None:
            rng = self.rng
            for epoch in range(self.cfg.train.num_epochs_hmc_bootstrap):
                rng, subkey = jax.random.split(rng)
                self.train_epoch(epoch_idx=epoch)
                self.save_model(epoch=epoch, step=0)
                if epoch % 20 == 0:
                    self.sample(
                        rng=subkey,
                        n=self.cfg.train.num_resampling_steps,
                        burn_in=self.cfg.train.resampling_burn_in,
                        parallel_chains=self.cfg.train.num_resampling_parallel_chains,
                        name=None,
                    )

        for epoch_bts in range(epoch, self.cfg.train.num_epochs):
            self.train_epoch(epoch_idx=epoch_bts)
            if epoch_bts % 10 == 0:
                self.save_model(epoch=epoch_bts, step=0)
                self.sample(
                    rng=subkey,
                    n=self.cfg.train.num_resampling_steps,
                    burn_in=self.cfg.train.resampling_burn_in,
                    parallel_chains=self.cfg.train.num_resampling_parallel_chains,
                    name=None,
                )

    def sample(self, rng, n, burn_in, parallel_chains, name):
        kernel_fn = jax.jit(
            lambda x: self.L_state.apply_fn({"params": self.L_state.params}, x)
        )
        logging.info("Sampling...")
        start_time = time.time()
        samples, ar, t = metropolis_hastings_with_momentum(
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
        logging.info(f"Sampling took {time.time() - start_time} seconds")

        if name is not None:
            name = self.cfg.figure_path / Path(name)

        # index = 0
        # fig = plot_logistic_regression_samples(
        #         samples[:self.cfg.log.samples_to_plot],
        #         num_chains=parallel_chains,
        #         index=index,
        #         name=self.cfg.figure_path / Path(f"samples_logistic_regression_{index}.png"),
        #         )
        # fig1 = plot_histograms_logistic_regression(
        #             samples[:self.cfg.log.samples_to_plot],
        #             index=index,
        #             name=self.cfg.figure_path / Path(f"histograms_logistic_regression_{index}.png"),
        #         )
        # fig2 = plot_histograms2d_logistic_regression(
        #             samples[:self.cfg.log.samples_to_plot],
        #             index=index,
        #             name=self.cfg.figure_path / Path(f"histograms2d_logistic_regression_{index}.png"),
        #         )
        # predictions = get_predictions(self.X, samples[:, : self.X.shape[1]])
        # logging.info(f"Accuracy: {np.mean(predictions == self.t.astype(int))}")

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

    def save_model(self, epoch, step):
        ckpt = {"L": self.L_state, "D": self.D_state}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        orbax_checkpointer.save(
            (
                Path(self.checkpoint_path) / Path(f"{epoch}_{step}")
            ).absolute(),  # os.path.join(self.checkpoint_path,f"{epoch}_{step}"),
            ckpt,
            save_args=save_args,
            force=self.cfg.overwrite,
        )

        # log cfg into checkpoint_path
        if epoch == 0 and step == 0:
            with open(os.path.join(self.checkpoint_path, "cfg.txt"), "w") as f:
                f.write(str(self.cfg))

    def load_model(self, epoch, step):
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        ckpt = orbax_checkpointer.restore(
            os.path.join(self.checkpoint_path, f"{epoch}_{step}")
        )
        self.L_state = ckpt["L"]
        self.D_state = ckpt["D"]


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
