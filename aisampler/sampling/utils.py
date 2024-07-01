import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns


def get_hamiltonian_density_image(
    density, xlim_q, ylim_q, xlim_p, ylim_p, q_0=0.0, q_1=1.0, n=100
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
    z_p = jnp.concatenate(
        jnp.array(
            [
                jnp.hstack([jnp.zeros((n**2, 1)) + q_0, jnp.zeros((n**2, 1)) + q_1]),
                jnp.hstack([X_p.reshape(-1, 1), Y_p.reshape(-1, 1)]),
            ]
        ),
        axis=1,
    )
    Z_q = jnp.exp(-density(z_q)).reshape((n, n))
    Z_p = jnp.exp(-density(z_p)).reshape((n, n))

    return Z_p, Z_q



def plot_samples_with_density(
    samples, target_density, q_0=0.0, q_1=0.0, name=None, ar=None, **kwargs
):
    xlim_q = 8  # jnp.max(jnp.abs(samples[:, 0])) + 1.5
    ylim_q = 8  # jnp.max(jnp.abs(samples[:, 1])) + 1.5
    xlim_p = 8  # jnp.max(jnp.abs(samples[:, 2])) + 1.5
    ylim_p = 8  # jnp.max(jnp.abs(samples[:, 3])) + 1.5

    Z_p, Z_q = get_hamiltonian_density_image(
        target_density, xlim_q, ylim_q, xlim_p, ylim_p, q_0=q_0, q_1=q_1, n=100
    )

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    if ar is not None:
        fig.suptitle(f"Acceptance rate: {ar:.3}", fontsize=25)
    
    contour_q = ax.contourf(jnp.linspace(-xlim_q, xlim_q, Z_q.shape[0]), jnp.linspace(-ylim_q, ylim_q, Z_q.shape[1]), Z_q, cmap="viridis")

    ax.scatter(samples[:, 0], samples[:, 1], c="red", alpha=0.6, s=10.5)
    ax.plot(samples[:, 0], samples[:, 1], **kwargs)

    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=25)

    plt.tight_layout()

    if name is not None:
        plt.savefig(name)
    plt.show()

    plt.close()

    return fig

def plot_kde(samples, name=None):
    # sns.set(style="whitegrid")
    plt.figure(figsize=(5, 5))
    sns.set(style="ticks")
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    sns.kdeplot(x=samples[:, 0], y=samples[:, 1], bw_adjust=.5, fill=True, cmap="viridis", shade_lowest=True)
    plt.tick_params(axis="x", labelsize=25)
    plt.tick_params(axis="y", labelsize=25)
    plt.tight_layout()
    if name is not None:
        plt.savefig(name)
    plt.show()


def plot_chain(samples, target_density, q_0=0.0, q_1=0.0, name=None, ar=None, **kwargs):
    xlim_q = jnp.max(jnp.abs(samples[:, 0])) + 1.5
    ylim_q = jnp.max(jnp.abs(samples[:, 1])) + 1.5
    xlim_p = jnp.max(jnp.abs(samples[:, 2])) + 1.5
    ylim_p = jnp.max(jnp.abs(samples[:, 3])) + 1.5

    _, Z_q = get_hamiltonian_density_image(
        target_density, xlim_q, ylim_q, xlim_p, ylim_p, q_0=q_0, q_1=q_1, n=100
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    if ar is not None:
        fig.suptitle(f"Acceptance rate: {ar:.3}")
    ax.imshow(Z_q, extent=(-xlim_q, xlim_q, -ylim_q, ylim_q), origin="lower", cmap="viridis")
    ax.scatter(samples[:, 0], samples[:, 1], s=0.5)
    ax.plot(samples[:, 0], samples[:, 1], **kwargs)
    ax.set_title("q")
    ax.set_xlabel("q1")
    ax.set_ylabel("q2")
    if name is not None:
        plt.savefig(name)
    plt.show()

    return fig


def get_density_image(density, xlim, ylim, d, n=100):
    x = jnp.linspace(-xlim, xlim, n)
    y = jnp.linspace(-ylim, ylim, n)
    X, Y = jnp.meshgrid(x, y)
    Z = jnp.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])
    Z = jnp.concatenate([Z, jnp.zeros((n**2, d - 2))], axis=1)
    Z = jnp.exp(-density(Z)).reshape((n, n))
    Z = Z.reshape(X.shape)

    return Z


def plot_samples(samples, target_density, d, name=None, ar=None, **kwargs):
    xlim = jnp.max(jnp.abs(samples[:, 0])) + 3.5
    ylim = jnp.max(jnp.abs(samples[:, 1])) + 3.5

    Z = get_density_image(target_density, xlim, ylim, d, n=100)

    plt.imshow(Z, extent=(-xlim, xlim, -ylim, ylim), origin="lower", cmap="viridis")
    plt.scatter(samples[:, 0], samples[:, 1], **kwargs)
    plt.xlabel("x1")
    plt.ylabel("x2")

    if ar is not None:
        plt.title(f"Acceptance rate: {ar:.3}")

    if name is not None:
        plt.savefig(name)
    plt.show()
