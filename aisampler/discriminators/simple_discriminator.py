import jax.numpy as jnp
import flax.linen as nn
from aisampler.kernels import create_henon_flow


class Discriminator(nn.Module):

    L: nn.Module
    D: nn.Module
    d: int

    def setup(self) -> None:

        self.R = jnp.concatenate(
            jnp.array([[1.0 for _ in range(self.d)], [-1.0 for _ in range(self.d)]])
        )

    def __call__(self, x):
        return self.D.psi(self.R * self.L(x) + x) * (
            self.D.eta(self.R * self.L(x)) - self.D.eta(x)
        )


class D(nn.Module):
    psi: nn.Module
    eta: nn.Module


def create_simple_discriminator(
    num_flow_layers: int,
    num_hidden_flow: int,
    num_layers_flow: int,
    num_layers_psi: int,
    num_hidden_psi: int,
    num_layers_eta: int,
    num_hidden_eta: int,
    activation: str,
    d: int,
) -> Discriminator:

    activation = getattr(nn, activation)

    return Discriminator(
        L=create_henon_flow(
            num_flow_layers=num_flow_layers,
            num_layers=num_layers_flow,
            num_hidden=num_hidden_flow,
            d=d,
        ),
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
