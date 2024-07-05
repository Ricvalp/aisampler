import jax
import jax.numpy as jnp


def get_sample_fn(
    kernel_fn: callable,
    density: callable,
    d: int,
    n: int,
    cov_p: jnp.ndarray,
    parallel_chains: int = 100,
    burn_in: int = 100,
    initial_std: float = 1.0,
    starting_points: jnp.ndarray = None,
    vstack: bool = True,
) -> callable:
    """
    Function to get the (jitted) Metropolis-Hastings with momentum sampler for a given kernel function and density.
    The kernel function takes in the current state and the parameters dictionary and returns the new stata.

    Args:
        kernel_fn (callable):
        density (callable):
        d (int):
        n (int):
        cov_p (jnp.ndarray):
        parallel_chains (int, optional): Defaults to 100.
        burn_in (int, optional): Defaults to 100.
        initial_std (float, optional): Defaults to 1.0.
        starting_points (jnp.ndarray, optional): Defaults to None.
        vstack (bool, optional): Defaults to True.

    Returns:
        callable:
    """

    def mh_kernel_with_momentum(
        x: jnp.ndarray,
        params: dict,
        key: jnp.ndarray,
    ) -> tuple:
        """ """

        key, accept_subkey, momentum_subkey = jax.random.split(key, 3)
        x_new = kernel_fn(x, params)

        log_prob_new = density(x_new)
        log_prob_old = density(x)
        log_prob_ratio = log_prob_old - log_prob_new

        accept = jax.random.uniform(accept_subkey, (parallel_chains,)) < jnp.exp(
            log_prob_ratio
        )

        x_new = jnp.where(accept[:, None], x_new, x)[:, : x.shape[1] // 2]
        momentum = jax.random.multivariate_normal(
            momentum_subkey, jnp.zeros(x.shape[1] // 2), cov_p, (parallel_chains,)
        )
        x_new = jnp.concatenate([x_new, momentum], axis=1)

        return x_new, accept.mean(), key

    jit_mh_kernel_with_momentum = jax.jit(mh_kernel_with_momentum)

    def metropolis_hastings_with_momentum(
        params: dict,
        rng: jnp.ndarray = jax.random.PRNGKey(42),
    ):
        first_init_subkey, second_init_subkey, sampling_subkey = jax.random.split(
            rng, 3
        )

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

        samples = []
        ars = []

        for i in range(n + burn_in):
            x, ar, sampling_subkey = jit_mh_kernel_with_momentum(
                x, params, sampling_subkey
            )
            if i >= burn_in:
                samples.append(x)
                ars.append(ar)
        if vstack:
            return jnp.vstack(samples), jnp.array(ars).mean()
        else:
            return jnp.transpose(jnp.array(samples), (1, 0, 2)), jnp.array(ars).mean()

    return metropolis_hastings_with_momentum
