import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import linen as nn


def plot_discriminator(
    discriminator,
    xlim_q,
    ylim_q,
    xlim_p,
    ylim_p,
    x_0,
    p_0,
    p_1,
    n=100,
    name=None,
):
    x = jnp.linspace(-xlim_q, xlim_q, n)
    y = jnp.linspace(-ylim_q, ylim_q, n)
    X_q, Y_q = jnp.meshgrid(x, y)
    x = jnp.linspace(-xlim_p, xlim_p, n)
    y = jnp.linspace(-ylim_p, ylim_p, n)
    X_p, Y_p = jnp.meshgrid(x, y)
    z_q = jnp.concatenate(
        jnp.array(
            [
                jnp.hstack([X_q.reshape(-1, 1), Y_q.reshape(-1, 1)]),
                jnp.hstack([jnp.zeros((n**2, 1)), jnp.zeros((n**2, 1))]),
            ]
        ),
        axis=1,
    )

    def r(y):
        return 1 / (1 + jnp.exp(-y))

    x_0 = jnp.array(
        [jnp.concatenate([x_0, jnp.array([p_0, p_1])], axis=0) for _ in range(n**2)]
    )
    ar = discriminator(x_0, z_q).reshape((n, n))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im = ax.imshow(
        r(ar), extent=(-xlim_q, xlim_q, -ylim_q, ylim_q), origin="lower", cmap="viridis"
    )
    ax.scatter(x_0[:, 0], x_0[:, 1], c="red", s=150.0, marker="x")

    ax.tick_params(axis="x", labelsize=25)  # Set x-axis tick font size
    ax.tick_params(axis="y", labelsize=25)  # Set y-axis tick font size

    # fig.colorbar(im, ax=ax)
    if name is not None:
        plt.savefig(f"{name}.png")
    plt.show()
    return fig


def get_discriminator_function(
    discriminator_parameters,
    num_layers_psi,
    num_hidden_psi,
    num_layers_eta,
    num_hidden_eta,
    activation,
    d,
):
    discriminator = create_simple_discriminator_without_kernel(
        num_layers_psi=num_layers_psi,
        num_hidden_psi=num_hidden_psi,
        num_layers_eta=num_layers_eta,
        num_hidden_eta=num_hidden_eta,
        activation=activation,
        d=d,
    )

    def discriminator_function(x, y):
        return discriminator.apply(discriminator_parameters, x, y)

    return discriminator_function


class DiscriminatorWithoutL(nn.Module):
    D: nn.Module
    d: int

    def setup(self) -> None:
        self.R = jnp.concatenate(
            jnp.array([[1.0 for _ in range(self.d)], [-1.0 for _ in range(self.d)]])
        )

    def __call__(self, x, y):
        return self.D.psi(self.R * y + x) * (self.D.eta(self.R * y) - self.D.eta(x))


class D(nn.Module):
    psi: nn.Module
    eta: nn.Module


def create_simple_discriminator_without_kernel(
    num_layers_psi: int,
    num_hidden_psi: int,
    num_layers_eta: int,
    num_hidden_eta: int,
    activation: str,
    d: int,
) -> DiscriminatorWithoutL:
    activation = getattr(nn, activation)

    return DiscriminatorWithoutL(
        D=D(
            psi=nn.Sequential(
                [nn.Dense(num_hidden_psi), activation]
                + [
                    nn.Dense(num_hidden_psi),
                    activation,
                ]
                * (num_layers_psi - 1)
                + [nn.Dense(1)]
            ),
            eta=nn.Sequential(
                [nn.Dense(num_hidden_eta), activation]
                + [
                    nn.Dense(num_hidden_eta),
                    activation,
                ]
                * (num_layers_eta - 1)
                + [nn.Dense(1)]
            ),
        ),
        d=d,
    )


def log_plot(
    discriminator_parameters,
    num_layers_psi: int,
    num_hidden_psi: int,
    num_layers_eta: int,
    num_hidden_eta: int,
    activation: str,
    d: int,
    name=None,
):
    discriminator_fn = get_discriminator_function(
        discriminator_parameters,
        num_layers_psi,
        num_hidden_psi,
        num_layers_eta,
        num_hidden_eta,
        activation,
        d,
    )

    fig = plot_discriminator(
        discriminator_fn,
        xlim_q=6,
        ylim_q=6,
        xlim_p=6,
        ylim_p=6,
        n=100,
        x_0=jnp.array([0.0, 0.0]),
        p_0=0.0,
        p_1=0.0,
        name=name,
    )

    return fig
