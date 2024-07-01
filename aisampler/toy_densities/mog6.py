import jax
import jax.numpy as jnp
from jax import grad


statistics_mog6 = {"mu": [0.0, 0.0], "sigma": [3.6, 3.6]}


def normal(x, mu, inv_cov):
    return jnp.exp(-0.5 * jnp.dot(jnp.dot((x - mu).T, inv_cov), (x - mu)))


normal = jax.vmap(normal, in_axes=(0, None, None))


def mog6(
    x,
    mu1=jnp.array([5.0, 0.0]),
    mu2=jnp.array([-5.0, 0.0]),
    mu3=jnp.array([0.0, 5.0]),
    inv_cov=jnp.eye(2) * 2,
):
    mus = [
        5 * jnp.array([jnp.sin(jnp.pi * i / 3), jnp.cos(jnp.pi * i / 3)])
        for i in range(6)
    ]
    return -jnp.log(
        (1 / 6) * normal(x, mus[0], inv_cov)
        + (1 / 6) * normal(x, mus[1], inv_cov)
        + (1 / 6) * normal(x, mus[2], inv_cov)
        + (1 / 6) * normal(x, mus[3], inv_cov)
        + (1 / 6) * normal(x, mus[4], inv_cov)
        + (1 / 6) * normal(x, mus[5], inv_cov)
    )


def hamiltonian_mog6(
    x,
    mu1=jnp.array([5.0, 0.0]),
    mu2=jnp.array([-5.0, 0.0]),
    mu3=jnp.array([0.0, 5.0]),
    inv_cov=jnp.eye(2) * 2,
    inv_cov_p=jnp.eye(2),
):
    d = x.shape[1]
    return mog6(x[:, : d // 2], mu1, mu2, mu3, inv_cov) - jnp.log(
        normal(x[:, d // 2 :], jnp.zeros(d // 2), inv_cov_p)
    )


def nv_normal(x, mu, inv_cov):
    return jnp.exp(-0.5 * jnp.dot(jnp.dot((x - mu).T, inv_cov), (x - mu)))


def nv_mog6(
    x,
    mu1=jnp.array([5.0, 0.0]),
    mu2=jnp.array([-5.0, 0.0]),
    mu3=jnp.array([0.0, 5.0]),
    inv_cov=jnp.eye(2) * 2,
):
    mus = [
        5 * jnp.array([jnp.sin(jnp.pi * i / 3), jnp.cos(jnp.pi * i / 3)])
        for i in range(6)
    ]
    return -jnp.log(
        (1 / 6) * nv_normal(x, mus[0], inv_cov)
        + (1 / 6) * nv_normal(x, mus[1], inv_cov)
        + (1 / 6) * nv_normal(x, mus[2], inv_cov)
        + (1 / 6) * nv_normal(x, mus[3], inv_cov)
        + (1 / 6) * nv_normal(x, mus[4], inv_cov)
        + (1 / 6) * nv_normal(x, mus[5], inv_cov)
    )


grad_mog6 = jax.vmap(grad(nv_mog6))


# def normal(x, mu, inv_cov):
#     return jnp.exp(-0.5 * jnp.dot(jnp.dot((x - mu).T, inv_cov), (x - mu)))

# normal = jax.vmap(normal, in_axes=(0, None, None))

# # def mog6(x):
# #     mus = [5 * jnp.array([jnp.sin(jnp.pi * i / 3), jnp.cos(jnp.pi * i / 3)]) for i in range(6)]
# #     inv_cov = jnp.eye(2) * 2
# #     return -jnp.log(jnp.sum(jnp.array([normal(x, mu, inv_cov) for mu in mus]), axis=0) / 6)

# def mog6(x):
#     mus = [5 * jnp.array([jnp.sin(jnp.pi * i / 3), jnp.cos(jnp.pi * i / 3)]) for i in range(6)]
#     inv_cov = jnp.eye(2) * 2
#     return -jnp.log(
#         normal(x, mus[0], inv_cov)/3 +
#         normal(x, mus[1], inv_cov)/3 +
#         normal(x, mus[2], inv_cov)/3
#         )

# def hamiltonian_mog6(x, inv_cov_p=jnp.eye(2) * 2):
#     d = x.shape[1]
#     return mog6(x[:, : d // 2]) - jnp.log(normal(x[:, d // 2 :], jnp.zeros(2), inv_cov_p))

# def nv_normal(x, mu, inv_cov):
#     return jnp.exp(-0.5 * jnp.dot(jnp.dot((x - mu).T, inv_cov), (x - mu)))

# # def nv_mog6(x):
# #     mus = [5 * jnp.array([jnp.sin(jnp.pi * i / 3), jnp.cos(jnp.pi * i / 3)]) for i in range(6)]
# #     inv_cov = jnp.eye(2) * 2
# #     return -jnp.log(jnp.sum(jnp.array([nv_normal(x, mu, inv_cov) for mu in mus]), axis=0) / 6)

# def nv_mog6(x):
#     mus = [5 * jnp.array([jnp.sin(jnp.pi * i / 3), jnp.cos(jnp.pi * i / 3)]) for i in range(6)]
#     inv_cov = jnp.eye(2) * 2
#     return -jnp.log(
#         nv_normal(x, mus[0], inv_cov)/3 +
#         nv_normal(x, mus[1], inv_cov)/3 +
#         nv_normal(x, mus[2], inv_cov)/3
#         )

# grad_mog6 = jax.vmap(grad(nv_mog6))


# import matplotlib.pyplot as plt
# import pymc3 as pm

# define your probability distribution
# import numpy as np
# def logprob(x, ivar):
#     logp = -0.5 * np.sum(ivar * x**2)
#     grad = -ivar * x
#     return logp, grad

# def logprob(x):
#     return nv_mog6(x).tolist(), np.array(grad(nv_mog6)(x)).astype(float)

# from pyhmc import hmc
# ivar = 1. / np.random.rand(5)
# # samples = hmc(logprob, x0=np.random.randn(5), args=(ivar,), n_samples=1000)
# samples = hmc(logprob, x0=np.random.randn(2), n_samples=1000)

# import corner  # pip install triangle_plot
# figure = corner.corner(samples)
# figure.savefig('triangle.png')


# x = jnp.linspace(-8, 8, 100)
# X, Y = jnp.meshgrid(x, x)
# Z = jnp.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])
# # grads = grad_mog6(Z)
# density = jnp.exp(-mog6(Z)).reshape((100, 100))
# plt.imshow(density)
# # plt.colorbar()
# # plt.quiver(Z[:, 0], Z[:, 1], grads[:, 0], grads[:, 1])
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# mus = np.array([5 * jnp.array([jnp.sin(jnp.pi * i / 3), jnp.cos(jnp.pi * i / 3)]) for i in range(6)])
# idxs = [i for i in range(len(mus))]
# idxs = np.random.choice(idxs, size=(100000,))
# mus = mus[idxs]
# samples = []
# for mu in mus:
#     samples.append(np.random.multivariate_normal(mu, np.eye(2)*0.5))

# samples = np.array(samples)
# print(f"std x: {np.std(samples[:, 0])}")
# print(f"std y: {np.std(samples[:, 1])}")
# plt.scatter(samples[:, 0], samples[:, 1])
# plt.show()
