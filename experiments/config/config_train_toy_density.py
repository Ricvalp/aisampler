from datetime import datetime
from ml_collections import ConfigDict


def get_config():

    cfg = ConfigDict()
    cfg.seed = 42

    cfg.figure_path = "./figures/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    # Target density
    cfg.target_density_name = "hamiltonian_ring"

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.checkpoint_dir = "./checkpoints"
    cfg.checkpoint.overwrite = True
    cfg.checkpoint.save_every = 50

    # Wandb
    cfg.wandb = ConfigDict()
    cfg.wandb.use = False
    cfg.wandb.project = "adversarial-involutive-sampler-toy-density"
    cfg.wandb.entity = "ricvalp"

    # Kernel
    cfg.kernel = ConfigDict()
    cfg.kernel.num_flow_layers = 5
    cfg.kernel.num_layers = 2
    cfg.kernel.num_hidden = 32
    cfg.kernel.d = 2

    # Discriminator
    cfg.discriminator = ConfigDict()
    cfg.discriminator.num_layers_psi = 3
    cfg.discriminator.num_hidden_psi = 128
    cfg.discriminator.num_layers_eta = 3
    cfg.discriminator.num_hidden_eta = 128
    cfg.discriminator.activation = "relu"

    # Train
    cfg.train = ConfigDict()
    cfg.train.init = "glorot_normal"
    cfg.train.kernel_learning_rate = 1e-4
    cfg.train.discriminator_learning_rate = 1e-4
    cfg.train.num_resampling_steps = 100
    cfg.train.num_resampling_parallel_chains = 500
    cfg.train.resampling_burn_in = 100
    cfg.train.batch_size = 4096
    cfg.train.num_epochs = 501
    cfg.train.num_AR_steps = 3
    cfg.train.num_adversarial_steps = 1

    return cfg
