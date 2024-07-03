import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns


def get_hamiltonian_density_image(density, xlim_q, ylim_q, xlim_p, ylim_p, n=100):
    x = jnp.linspace(-xlim_q, xlim_q, n)
    y = jnp.linspace(-ylim_q, ylim_q, n)
    X_q, Y_q = jnp.meshgrid(x, y)
    x = jnp.linspace(-xlim_p, xlim_p, n)
    y = jnp.linspace(-ylim_p, ylim_p, n)
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

    return Z_q


def plot_samples_with_density(
    samples,
    target_density,
    name=None,
    ar=None,
    include_trajectories=False,
    **kwargs,
):
    xlim_q = 8  # jnp.max(jnp.abs(samples[:, 0])) + 1.5
    ylim_q = 8  # jnp.max(jnp.abs(samples[:, 1])) + 1.5
    xlim_p = 8  # jnp.max(jnp.abs(samples[:, 2])) + 1.5
    ylim_p = 8  # jnp.max(jnp.abs(samples[:, 3])) + 1.5

    Z_q = get_hamiltonian_density_image(
        target_density, xlim_q, ylim_q, xlim_p, ylim_p, n=100
    )

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    if ar is not None:
        fig.suptitle(f"Acceptance rate: {ar:.3}", fontsize=25)

    contour_q = ax.contourf(
        jnp.linspace(-xlim_q, xlim_q, Z_q.shape[0]),
        jnp.linspace(-ylim_q, ylim_q, Z_q.shape[1]),
        Z_q,
        cmap="viridis",
    )

    ax.scatter(samples[:, 0], samples[:, 1], c="red", alpha=0.4, s=1.5, **kwargs)

    if include_trajectories:
        ax.plot(samples[:, 0], samples[:, 1])

    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=25)

    ax.set_xlim(-xlim_q, xlim_q)
    ax.set_ylim(-ylim_q, ylim_q)

    plt.tight_layout()

    if name is not None:
        plt.savefig(name)
    plt.show()

    plt.close()

    return fig


def plot_kde(samples, name=None):
    plt.figure(figsize=(5, 5))
    sns.set(style="ticks")
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    sns.kdeplot(
        x=samples[:, 0],
        y=samples[:, 1],
        bw_adjust=0.5,
        fill=True,
        cmap="viridis",
        thresh=0,
    )
    plt.tick_params(axis="x", labelsize=25)
    plt.tick_params(axis="y", labelsize=25)
    plt.tight_layout()
    if name is not None:
        plt.savefig(name)
    plt.show()
