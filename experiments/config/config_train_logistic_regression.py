from datetime import datetime
from ml_collections import ConfigDict


def get_config():

    cfg = ConfigDict()
    cfg.seed = 42

    cfg.figure_path = "./figures/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.checkpoint_dir = "./checkpoints"
    cfg.checkpoint.overwrite = True
    cfg.checkpoint.save_best_only = True
    cfg.checkpoint.save_every = 1

    cfg.dataset_name = "Heart"

    cfg.wandb = ConfigDict()
    cfg.wandb.use = False
    cfg.wandb.project = "adversarial-involutive-sampler-loggistic-regression"
    cfg.wandb.entity = "ricvalp"

    cfg.kernel = ConfigDict()
    cfg.kernel.num_flow_layers = 5
    cfg.kernel.num_layers = 2
    cfg.kernel.num_hidden = 64

    cfg.discriminator = ConfigDict()
    cfg.discriminator.num_layers_psi = 3
    cfg.discriminator.num_hidden_psi = 256
    cfg.discriminator.num_layers_eta = 3
    cfg.discriminator.num_hidden_eta = 256
    cfg.discriminator.activation = "relu"

    cfg.train = ConfigDict()
    cfg.train.kernel_learning_rate = 1e-4
    cfg.train.discriminator_learning_rate = 1e-4
    cfg.train.num_resampling_steps = 500
    cfg.train.num_resampling_parallel_chains = 100
    cfg.train.resampling_burn_in = 100
    cfg.train.batch_size = 8192
    cfg.train.num_epochs = 200
    cfg.train.num_AR_steps = 3
    cfg.train.num_adversarial_steps = 1

    cfg.hmc_bootstrapping = ConfigDict()
    cfg.hmc_bootstrapping.use = True
    cfg.hmc_bootstrapping.num_epochs = 100
    cfg.hmc_bootstrapping.num_steps = 40
    cfg.hmc_bootstrapping.step_size = 0.05
    cfg.hmc_bootstrapping.num_parallel_chains = 50
    cfg.hmc_bootstrapping.num_iterations = 1000
    cfg.hmc_bootstrapping.burn_in = 1000
    cfg.hmc_bootstrapping.initial_std = 0.1
    cfg.hmc_bootstrapping.seed = 42

    return cfg
