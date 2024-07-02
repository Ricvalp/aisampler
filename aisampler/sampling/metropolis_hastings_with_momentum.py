import time

import jax
import jax.numpy as jnp
from absl import logging


def mh_kernel_with_momentum(
    x: jnp.ndarray,
    key: jnp.ndarray,
    cov_p: jnp.ndarray,
    kernel: callable,
    density: callable,
    parallel_chains: int = 100,
) -> tuple:
    """
    Metropolis-Hastings kernel with momentum for a given kernel and density.

    Args:

        x: jnp.ndarray
        key: jnp.ndarray
        cov_p: jnp.ndarray
        kernel: callable
        density: callable
        parallel_chains: int
    Returns:

        tuple: Tuple of the new samples, acceptance rate, and the key.
    """

    key, accept_subkey, momentum_subkey = jax.random.split(key, 3)
    x_new = kernel(x)

    log_prob_new = density(x_new)
    log_prob_old = density(x)
    log_prob_ratio = log_prob_old - log_prob_new  # log_prob_old - log_prob_new

    accept = jax.random.uniform(accept_subkey, (parallel_chains,)) < jnp.exp(
        log_prob_ratio
    )

    # accept = (
    #     jnp.log(jax.random.uniform(accept_subkey, (parallel_chains,))) < log_prob_ratio
    # )

    x_new = jnp.where(accept[:, None], x_new, x)[:, : x.shape[1] // 2]
    momentum = jax.random.multivariate_normal(
        momentum_subkey, jnp.zeros(x.shape[1] // 2), cov_p, (parallel_chains,)
    )
    x_new = jnp.concatenate([x_new, momentum], axis=1)

    return x_new, accept.mean(), key


jit_mh_kernel_with_momentum = jax.jit(
    mh_kernel_with_momentum,
    static_argnums=(
        3,
        4,
        5,
    ),
)


def metropolis_hastings_with_momentum(
    kernel: callable,
    density: callable,
    d: int,
    n: int,
    cov_p: jnp.ndarray,
    parallel_chains: int = 100,
    burn_in: int = 100,
    rng: jnp.ndarray = jax.random.PRNGKey(42),
    initial_std: float = 1.0,
    starting_points: jnp.ndarray = None,
    vstack: bool = False,
) -> tuple:
    """

    Metropolis-Hastings with momentum for a given kernel and density.

    Args:
        kernel: callable
        density: callable
        d: int
        n: int
        cov_p: jnp.ndarray
        parallel_chains: int
        burn_in: int
        rng: jnp.ndarray
        initial_std: float
        starting_points: jnp.ndarray
        vstack: bool
    Returns:
        tuple: Tuple of samples and acceptance rate.
    """
    first_init_subkey, second_init_subkey, sampling_subkey = jax.random.split(rng, 3)

    if starting_points is None:
        x = jax.random.normal(first_init_subkey, (parallel_chains, d)) * initial_std
        x = jnp.concatenate(
            [
                x,
                jax.random.multivariate_normal(
                    second_init_subkey, jnp.zeros(d), cov_p, (parallel_chains,)
                ),
            ],
            axis=1,
        )
    else:
        x = starting_points[:parallel_chains]

    jit_mh_kernel_with_momentum(
        x, sampling_subkey, cov_p, kernel, density, parallel_chains=parallel_chains
    )

    samples = []
    ars = []

    for i in range(n + burn_in):
        x, ar, sampling_subkey = jit_mh_kernel_with_momentum(
            x, sampling_subkey, cov_p, kernel, density, parallel_chains=parallel_chains
        )
        if i >= burn_in:
            samples.append(x)
            ars.append(ar)
    if vstack:
        return jnp.vstack(samples), jnp.array(ars).mean()
    else:
        return jnp.transpose(jnp.array(samples), (1, 0, 2)), jnp.array(ars).mean()


@jax.jit
def jitted_stack(x):
    return jnp.stack(x)
