import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns


# def plot_hamiltonian_density(
#     density, xlim_q, ylim_q, xlim_p, ylim_p, n=100, q_0=0.0, q_1=0.0, name=None
# ):
#     x = jnp.linspace(-xlim_q, xlim_q, n)
#     y = jnp.linspace(-ylim_q, ylim_q, n)
#     X_q, Y_q = jnp.meshgrid(x, y)
#     x = jnp.linspace(-xlim_p, ylim_p, n)
#     y = jnp.linspace(-ylim_p, ylim_p, n)
#     X_p, Y_p = jnp.meshgrid(x, y)
#     z_q = jnp.concatenate(
#         jnp.array(
#             [
#                 jnp.hstack([X_q.reshape(-1, 1), Y_q.reshape(-1, 1)]),
#                 jnp.hstack([jnp.zeros((n**2, 1)), jnp.zeros((n**2, 1))]),
#             ]
#         ),
#         axis=1,
#     )
#     z_p = jnp.concatenate(
#         jnp.array(
#             [
#                 jnp.hstack([jnp.zeros((n**2, 1)) + q_0, jnp.zeros((n**2, 1)) + q_1]),
#                 jnp.hstack([X_p.reshape(-1, 1), Y_p.reshape(-1, 1)]),
#             ]
#         ),
#         axis=1,
#     )
#     Z_q = jnp.exp(-density(z_q)).reshape((n, n))
#     Z_p = jnp.exp(-density(z_p)).reshape((n, n))

#     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#     im_q = ax[0].imshow(
#         Z_q, extent=(-xlim_q, xlim_q, -ylim_q, ylim_q), origin="lower", cmap="viridis"
#     )
#     ax[0].set_title("q")
#     ax[0].set_xlabel("q1")
#     ax[0].set_ylabel("q2")
#     fig.colorbar(im_q, ax=ax[0])
#     im_p = ax[1].imshow(
#         Z_p, extent=(-xlim_p, xlim_p, -ylim_p, ylim_p), origin="lower", cmap="viridis"
#     )
#     ax[1].set_title("p")
#     ax[1].set_xlabel("p1")
#     ax[1].set_ylabel("p2")
#     fig.colorbar(im_p, ax=ax[1])

#     if name is not None:
#         plt.savefig(name)
#     plt.show()
#     plt.close()


def plot_hamiltonian_density(density, xlim_q=-8.0, ylim_q=-8.0, n=200, name=None):

    x = jnp.linspace(-xlim_q, xlim_q, n)
    y = jnp.linspace(-ylim_q, ylim_q, n)
    X_q, Y_q = jnp.meshgrid(x, y)
    z_q = jnp.concatenate(
        jnp.array(
            [
                jnp.hstack([X_q.reshape(-1, 1), Y_q.reshape(-1, 1)]),
                jnp.hstack([jnp.zeros((n**2, 1)), jnp.zeros((n**2, 1))]),
            ]
        ),
        axis=1,
    )
    Z_q = jnp.exp(-density(z_q)).reshape((n, n))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    contour_q = ax.contourf(
        jnp.linspace(-xlim_q, xlim_q, Z_q.shape[0]),
        jnp.linspace(-ylim_q, ylim_q, Z_q.shape[1]),
        Z_q,
        cmap="viridis",
    )

    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=25)

    plt.tight_layout()

    if name is not None:
        plt.savefig(name)
    plt.show()
    plt.close()


# def plot_logistic_regression_density(
#     density, xlim_q, ylim_q, xlim_p, ylim_p, d, n=100, name=None
# ):
#     x = jnp.linspace(-xlim_q, xlim_q, n)
#     y = jnp.linspace(-ylim_q, ylim_q, n)
#     X, Y = jnp.meshgrid(x, y)

#     z = jnp.concatenate(
#         jnp.array(
#             [
#                 jnp.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)]),
#                 jnp.hstack([jnp.zeros((n**2, 1)), jnp.zeros((n**2, 1))]),
#             ]
#         ),
#         axis=1,
#     )

#     z = jnp.concatenate([z, jnp.zeros((n**2, d - 4))], axis=1)
#     Z = density(z).reshape((n, n))

#     fig = plt.figure(figsize=(10, 5))
#     plt.imshow(
#         Z, extent=(-xlim_q, xlim_q, -ylim_q, ylim_q), origin="lower", cmap="viridis"
#     )
#     plt.title("w")
#     plt.xlabel("w1")
#     plt.ylabel("w2")
#     plt.colorbar()

#     if name is not None:
#         plt.savefig(name)
#     plt.show()
#     plt.close()
