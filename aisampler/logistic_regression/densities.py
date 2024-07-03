import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import grad
from jax.scipy.special import expit


def sigma(x):
    return expit(x)


def normal(x, inv_sigma):
    return 0.5 * x @ inv_sigma @ x.T


vmap_normal = jax.vmap(normal, in_axes=(0, None))


def U(b, t, X, inv_sigma, eps):
    return -jnp.sum(
        t[:, None] * jnp.log(sigma(X @ b.T) + eps)
        + (1 - t[:, None]) * jnp.log(1 - sigma(X @ b.T) + eps),
        axis=0,
    ) + vmap_normal(b, inv_sigma)


def hamiltonian(x, t, X, inv_sigma):
    d = x.shape[-1] // 2
    b = x[:, :d]
    p = x[:, d:]
    return U(b, t, X, inv_sigma, 1e-10) + vmap_normal(p, jnp.eye(d) * 10)


def u(b, t, X, inv_sigma):
    return -jnp.sum(
        t * jnp.log(sigma(X @ b.T) + 1e-10)
        + (1 - t) * jnp.log(1 - sigma(X @ b.T) + 1e-10),
        axis=0,
    ) + normal(b, inv_sigma)


grad_U = jax.vmap(grad(u, argnums=0), in_axes=(0, None, None, None))


def logistic_function(x, w):
    return 1 - expit(x @ w)


def generate_dataset(n, w, rng):
    x = jnp.linspace(-1, 1, n)
    y = jnp.linspace(-1, 1, n)
    x, y = jnp.meshgrid(x, y)
    X = jnp.stack([x, y], axis=-1).reshape(n * n, 2)
    X = jnp.concatenate([X, jnp.ones((n * n, 1))], axis=-1)

    p = logistic_function(X, w)

    t = jax.random.bernoulli(rng, p=p)

    return t, X


def normalize_covariates(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


def get_predictions(X, samples):
    predictions = sigma(X @ samples.T).mean(axis=1)
    return (predictions > 0.5).astype(int)
