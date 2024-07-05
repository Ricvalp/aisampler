import os
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

try:
    import wandb
except ImportError:
    wandb = None

from aisampler.discriminators import create_simple_discriminator
from aisampler.sampling import (
    get_sample_fn,
)

from aisampler.sampling import hmc

from aisampler.trainers.utils import SamplesDataset, numpy_collate
from aisampler.sampling.metrics import ess


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
        self.create_sample_fn()

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

        samples, ar = self.sample_fn(
            self.L_state.params,
            rng=subkey,
        )

        dataset = SamplesDataset(np.array(samples))
        self.data_loader = DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            collate_fn=numpy_collate,
        )

        return key, ar

    def create_sample_fn(self):

        self.sample_fn = get_sample_fn(
            kernel_fn=lambda x, params: self.L_state.apply_fn({"params": params}, x),
            density=self.density,
            d=self.cfg.kernel.d,
            n=self.cfg.train.num_resampling_steps,
            cov_p=jnp.eye(self.cfg.kernel.d),
            parallel_chains=self.cfg.train.num_resampling_parallel_chains,
            burn_in=self.cfg.train.resampling_burn_in,
            initial_std=1.0,
            starting_points=None,
        )

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
        for epoch in tqdm(range(1, self.cfg.train.num_epochs)):
            ar_loss, adv_loss, ar = self.train_epoch(epoch_idx=epoch)

            tqdm.write(
                f"Epoch {epoch}: ar_loss={ar_loss:.4f}, adv_loss={adv_loss:.4f}, ar={ar:.4f}"
            )

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
        samples, ar = self.sample_fn(self.L_state.params, subkey)

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
            ar = -1.0
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

        return key, ar

    def create_sample_fn(self):

        self.sample_fn = get_sample_fn(
            kernel_fn=lambda x, params: self.L_state.apply_fn({"params": params}, x),
            density=self.density,
            d=self.density.dim,
            n=100,
            cov_p=jnp.eye(self.density.dim),
            parallel_chains=self.cfg.train.num_resampling_parallel_chains,
            burn_in=self.cfg.train.resampling_burn_in,
            initial_std=1.0,
            starting_points=self.hmc_samples,
        )

        self.logging_sample_fn = get_sample_fn(
            kernel_fn=lambda x, params: self.L_state.apply_fn({"params": params}, x),
            density=self.density,
            d=self.density.dim,
            n=100,
            cov_p=jnp.eye(self.density.dim),
            parallel_chains=self.cfg.train.num_resampling_parallel_chains,
            burn_in=50,
            initial_std=1.0,
            starting_points=self.hmc_samples,
        )

    def bootstrap_with_hmc(self, epoch_idx):
        if epoch_idx == 0:
            self.rng, _ = self.create_data_loader_with_hmc(self.rng)
            ar = -1.0
        else:
            self.rng, subkey = jax.random.split(self.rng)
            _, ar = self.logging_sample_fn(self.L_state.params, subkey)

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
                        "acceptance rate": ar,
                    }
                )

        return jnp.array(ar_losses).mean(), jnp.array(adv_losses).mean(), ar

    def train_epoch(self, epoch_idx):

        self.rng, ar = self.create_data_loader(self.rng)

        if epoch_idx % self.cfg.checkpoint.save_every == 0:
            if ar > self.best_ar:
                self.best_ar = ar
                self.save_model(epoch=epoch_idx)

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
                        "acceptance rate": ar,
                    }
                )

        return jnp.array(ar_losses).mean(), jnp.array(adv_losses).mean(), ar

    def train_model(self):

        self.best_ar = -1.0

        self.create_sample_fn()

        if self.cfg.hmc_bootstrapping.use:
            logging.info("Bootstrapping with HMC...")
            for epoch in tqdm(range(0, self.cfg.hmc_bootstrapping.num_epochs)):
                ar_loss, adv_loss, ar = self.bootstrap_with_hmc(epoch_idx=epoch)
                tqdm.write(
                    f"Epoch {epoch}: ar_loss={ar_loss:.4f}, adv_loss={adv_loss:.4f}, ar={ar:.4f}"
                )

        logging.info("Training model...")
        for epoch in tqdm(range(0, self.cfg.train.num_epochs)):
            ar_loss, adv_loss, ar = self.train_epoch(epoch_idx=epoch)

            tqdm.write(
                f"Epoch {epoch}: ar_loss={ar_loss:.4f}, adv_loss={adv_loss:.4f}, ar={ar:.4f}"
            )

    def save_model(self, epoch):
        if self.cfg.checkpoint.save_best_only:
            ckpt = {"L": self.L_state, "D": self.D_state}
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            save_args = orbax_utils.save_args_from_target(ckpt)
            orbax_checkpointer.save(
                (Path(self.checkpoint_path) / "best").absolute(),
                ckpt,
                save_args=save_args,
                force=self.cfg.checkpoint.overwrite,
            )
        else:
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
