import time

import jax
import jax.numpy as jnp
from absl import logging


def mh_kernel_with_momentum(x, key, cov_p, kernel, density, parallel_chains=100):
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
    kernel,
    density,
    d,
    n,
    cov_p,
    parallel_chains=100,
    burn_in=100,
    rng=jax.random.PRNGKey(42),
    initial_std=1.0,
    starting_points=None,
):
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

    # logging.info("Jitting AI kernel...")
    # time_start = time.time()

    jit_mh_kernel_with_momentum(
        x, sampling_subkey, cov_p, kernel, density, parallel_chains=parallel_chains
    )

    # logging.info(f"Jitting done. Time taken: {time.time() - time_start}")

    samples = []
    ars = []

    # logging.info("Sampling...")
    # time_start = time.time()

    for i in range(n + burn_in):
        x, ar, sampling_subkey = jit_mh_kernel_with_momentum(
            x, sampling_subkey, cov_p, kernel, density, parallel_chains=parallel_chains
        )
        if i >= burn_in:
            samples.append(x)
            ars.append(ar)

    # t = time.time() - time_start
    # logging.info(f"Sampling done. Time taken: {t}")

    # return jnp.transpose(jnp.array(samples), (1, 0, 2)), jnp.array(ars).mean() # , t

    return jnp.vstack(samples), jnp.array(ars).mean()


@jax.jit
def jitted_stack(x):
    return jnp.stack(x)
