from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn


default_init = nn.initializers.glorot_normal()


class FlowModel(nn.Module):
    d: int
    flows: Sequence[nn.Module]

    def setup(self):
        self.R = jax.numpy.array(
            [1.0 for i in range(self.d)] + [-1.0 for i in range(self.d)]
        )

    def __call__(self, x):
        z = x

        for flow in self.flows:
            z = flow(z, reverse=False)

        z = z * self.R

        for flow in reversed(self.flows):
            z = flow(z, reverse=True)

        z = z * self.R

        return z


class HenonLayer(nn.Module):
    V: nn.Module
    d: int

    def setup(self):
        self.eta = self.param("eta", nn.initializers.zeros, (1, self.d * 2))

        self.i1 = jnp.block(
            [
                [jnp.zeros((self.d, self.d)), jnp.eye((self.d))],
                [jnp.zeros((self.d, self.d)), jnp.zeros((self.d, self.d))],
            ]
        )

        self.i2 = jnp.block(
            [
                [jnp.zeros((self.d, self.d)), jnp.zeros((self.d, self.d))],
                [jnp.eye((self.d)), jnp.zeros((self.d, self.d))],
            ]
        )

        self.i3 = jnp.block(
            [
                [jnp.eye((self.d)), jnp.zeros((self.d, self.d))],
                [jnp.zeros((self.d, self.d)), jnp.zeros((self.d, self.d))],
            ]
        )

        self.i4 = jnp.block(
            [
                [jnp.zeros((self.d, self.d)), jnp.zeros((self.d, self.d))],
                [jnp.zeros((self.d, self.d)), jnp.eye((self.d))],
            ]
        )

    def __call__(self, z, reverse=False):
        if not reverse:
            X = jnp.matmul(z, self.i1)
            Y = jnp.matmul(z, self.i2)
            ETA = jnp.matmul(self.eta, self.i3)
            V = jnp.matmul(self.V(Y), self.i4)
            return -X + Y + ETA + V

        else:
            X = jnp.matmul(z, self.i1)
            Y = jnp.matmul(z, self.i2)
            ETA = jnp.matmul(self.eta, self.i4)
            Xbar = jnp.matmul(X - ETA, self.i2)
            V = jnp.matmul(self.V(Xbar), self.i2)
            return X - Y - ETA + V


class SimpleMLP(nn.Module):
    num_hidden: int
    num_layers: int
    num_outputs: int

    def setup(self):
        self.linears = [
            nn.Dense(features=self.num_hidden, kernel_init=default_init)
            for i in range(self.num_layers - 1)
        ] + [nn.Dense(features=self.num_outputs, kernel_init=default_init)]

    def __call__(self, x):
        for linear in self.linears[:-1]:
            x = linear(x)
            x = nn.relu(x)
        x = self.linears[-1](x)
        return x


def create_henon_flow(
    num_flow_layers: int, num_layers: int, num_hidden: int, d: int
) -> FlowModel:
    """
    Args:
        num_flow_layers: int
        num_layers: int
        num_hidden: int
        d: int
    Returns:
        FlowModel
    """
    flow_layers = []

    flow_layers += [
        HenonLayer(
            SimpleMLP(num_layers=num_layers, num_hidden=num_hidden, num_outputs=2 * d),
            d=d,
        )
        for _ in range(num_flow_layers)
    ]

    flow_model = FlowModel(d, flow_layers)

    return flow_model
