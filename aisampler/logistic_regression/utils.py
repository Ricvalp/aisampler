import matplotlib.pyplot as plt


def plot_logistic_regression_samples(
    samples, plot=plt.scatter, index=0, name=None, **kwargs
):

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0, 0].scatter(samples[:, 0 + index], samples[:, 1 + index], s=1, **kwargs)
    # axs[0, 0].scatter(
    #     samples[0, 0 + index],
    #     samples[0, 1 + index],
    #     s=10,
    #     c="red",
    #     marker="x",
    #     label="start",
    # )
    # axs[0, 0].scatter(
    #     samples[-1, 0 + index],
    #     samples[-1, 1 + index],
    #     s=10,
    #     c="green",
    #     marker="x",
    #     label="end",
    # )

    axs[0, 1].scatter(samples[:, 1 + index], samples[:, 2 + index], s=1, **kwargs)
    # axs[0, 1].scatter(
    #     samples[0, 1 + index],
    #     samples[0, 2 + index],
    #     s=10,
    #     c="red",
    #     marker="x",
    #     label="start",
    # )
    # axs[0, 1].scatter(
    #     samples[-1, 1 + index],
    #     samples[-1, 2 + index],
    #     s=10,
    #     c="green",
    #     marker="x",
    #     label="end",
    # )

    axs[1, 0].scatter(samples[:, 2 + index], samples[:, 3 + index], s=1, **kwargs)
    # axs[1, 0].scatter(
    #     samples[0, 2 + index],
    #     samples[0, 3 + index],
    #     s=10,
    #     c="red",
    #     marker="x",
    #     label="start",
    # )
    # axs[1, 0].scatter(
    #     samples[-1, 2 + index],
    #     samples[-1, 3 + index],
    #     s=10,
    #     c="green",
    #     marker="x",
    #     label="end",
    # )

    axs[1, 1].scatter(samples[:, 3 + index], samples[:, 4 + index], s=1, **kwargs)
    # axs[1, 1].scatter(
    #     samples[0, 3 + index],
    #     samples[0, 4 + index],
    #     s=10,
    #     c="red",
    #     marker="x",
    #     label="start",
    # )
    # axs[1, 1].scatter(
    #     samples[-1, 3 + index],
    #     samples[-1, 4 + index],
    #     s=10,
    #     c="green",
    #     marker="x",
    #     label="end",
    # )

    # if num_chains is not None:
    #     for n in range(num_chains - 1):
    #         axs[0, 0].scatter(
    #             samples[n, 0 + index], samples[n, 1 + index], s=10, c="red", marker="x"
    #         )
    #         axs[0, 0].scatter(
    #             samples[-n, 0 + index],
    #             samples[-n, 1 + index],
    #             s=10,
    #             c="green",
    #             marker="x",
    #         )
    #         axs[0, 1].scatter(
    #             samples[n, 1 + index], samples[n, 2 + index], s=10, c="red", marker="x"
    #         )
    #         axs[0, 1].scatter(
    #             samples[-n, 1 + index],
    #             samples[-n, 2 + index],
    #             s=10,
    #             c="green",
    #             marker="x",
    #         )
    #         axs[1, 0].scatter(
    #             samples[n, 2], samples[n, 3 + index], s=10, c="red", marker="x"
    #         )
    #         axs[1, 0].scatter(
    #             samples[-n, 2 + index],
    #             samples[-n, 3 + index],
    #             s=10,
    #             c="green",
    #             marker="x",
    #         )
    #         axs[1, 1].scatter(
    #             samples[n, 3 + index], samples[n, 4 + index], s=10, c="red", marker="x"
    #         )
    #         axs[1, 1].scatter(
    #             samples[-n, 3 + index],
    #             samples[-n, 4 + index],
    #             s=10,
    #             c="green",
    #             marker="x",
    #         )
    #     axs[0, 0].scatter(
    #         samples[n + 1, 0 + index],
    #         samples[n + 1, 1 + index],
    #         s=10,
    #         c="red",
    #         marker="x",
    #         label="start",
    #     )
    #     axs[0, 0].scatter(
    #         samples[-n - 1, 0 + index],
    #         samples[-n - 1, 1 + index],
    #         s=10,
    #         c="green",
    #         marker="x",
    #         label="end",
    #     )
    #     axs[0, 1].scatter(
    #         samples[n + 1, 1 + index],
    #         samples[n + 1, 2 + index],
    #         s=10,
    #         c="red",
    #         marker="x",
    #         label="start",
    #     )
    #     axs[0, 1].scatter(
    #         samples[-n - 1, 1 + index],
    #         samples[-n - 1, 2 + index],
    #         s=10,
    #         c="green",
    #         marker="x",
    #         label="end",
    #     )
    #     axs[1, 0].scatter(
    #         samples[n + 1, 2 + index],
    #         samples[n + 1, 3 + index],
    #         s=10,
    #         c="red",
    #         marker="x",
    #         label="start",
    #     )
    #     axs[1, 0].scatter(
    #         samples[-n - 1, 2 + index],
    #         samples[-n - 1, 3 + index],
    #         s=10,
    #         c="green",
    #         marker="x",
    #         label="end",
    #     )
    #     axs[1, 1].scatter(
    #         samples[n + 1, 3 + index],
    #         samples[n + 1, 4 + index],
    #         s=10,
    #         c="red",
    #         marker="x",
    #         label="start",
    #     )
    #     axs[1, 1].scatter(
    #         samples[-n - 1, 3 + index],
    #         samples[-n - 1, 4 + index],
    #         s=10,
    #         c="green",
    #         marker="x",
    #         label="end",
    #     )

    axs[0, 0].set_xlabel(f"$w_{index + 0}$")
    axs[0, 0].set_ylabel(f"$w_{index + 1}$")
    axs[0, 1].set_xlabel(f"$w_{index + 1}$")
    axs[0, 1].set_ylabel(f"$w_{index + 2}$")
    axs[1, 0].set_xlabel(f"$w_{index + 2}$")
    axs[1, 0].set_ylabel(f"$w_{index + 3}$")
    axs[1, 1].set_xlabel(f"$w_{index + 3}$")
    axs[1, 1].set_ylabel(f"$w_{index + 4}$")

    plt.show()

    if name is not None:
        plt.savefig(name)
    plt.close()

    return fig


def plot_histograms2d_logistic_regression(samples, index=0, name=None, **kwargs):

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0, 0].hist2d(samples[:, 0 + index], samples[:, 1 + index], bins=100, **kwargs)
    axs[0, 1].hist2d(samples[:, 1 + index], samples[:, 2 + index], bins=100, **kwargs)
    axs[1, 0].hist2d(samples[:, 2 + index], samples[:, 3 + index], bins=100, **kwargs)
    axs[1, 1].hist2d(samples[:, 3 + index], samples[:, 4 + index], bins=100, **kwargs)

    axs[0, 0].set_xlabel(f"$w_{index + 0}$")
    axs[0, 0].set_ylabel(f"$w_{index + 1}$")
    axs[0, 1].set_xlabel(f"$w_{index + 1}$")
    axs[0, 1].set_ylabel(f"$w_{index + 2}$")
    axs[1, 0].set_xlabel(f"$w_{index + 2}$")
    axs[1, 0].set_ylabel(f"$w_{index + 3}$")
    axs[1, 1].set_xlabel(f"$w_{index + 3}$")
    axs[1, 1].set_ylabel(f"$w_{index + 4}$")

    plt.show()

    if name is not None:
        plt.savefig(name)
    plt.close()

    return fig


def plot_histograms_logistic_regression(samples, index=0, name=None, **kwargs):

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0, 0].hist(samples[:, 0 + index], bins=100, density=True, **kwargs)
    axs[0, 1].hist(samples[:, 1 + index], bins=100, density=True, **kwargs)
    axs[1, 0].hist(samples[:, 2 + index], bins=100, density=True, **kwargs)
    axs[1, 1].hist(samples[:, 3 + index], bins=100, density=True, **kwargs)

    axs[0, 0].set_xlabel(f"$w_{index + 0}$")
    axs[0, 1].set_xlabel(f"$w_{index + 1}$")
    axs[1, 0].set_xlabel(f"$w_{index + 2}$")
    axs[1, 1].set_xlabel(f"$w_{index + 3}$")

    plt.show()

    if name is not None:
        plt.savefig(name)
    plt.close()

    return fig


def plot_first_kernel_iteration(kernel, starting_points, index=0, name=None):

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    samples = kernel(starting_points)

    axs[0, 0].scatter(
        starting_points[:, 0 + index], starting_points[:, 1 + index], s=1, c="b"
    )
    axs[0, 1].scatter(
        starting_points[:, 1 + index], starting_points[:, 2 + index], s=1, c="b"
    )
    axs[1, 0].scatter(
        starting_points[:, 2 + index], starting_points[:, 3 + index], s=1, c="b"
    )
    axs[1, 1].scatter(
        starting_points[:, 3 + index], starting_points[:, 4 + index], s=1, c="b"
    )

    axs[0, 0].scatter(
        samples[:, 0 + index], samples[:, 1 + index], s=1, c="r", alpha=0.4
    )
    axs[0, 1].scatter(
        samples[:, 1 + index], samples[:, 2 + index], s=1, c="r", alpha=0.4
    )
    axs[1, 0].scatter(
        samples[:, 2 + index], samples[:, 3 + index], s=1, c="r", alpha=0.4
    )
    axs[1, 1].scatter(
        samples[:, 3 + index], samples[:, 4 + index], s=1, c="r", alpha=0.4
    )

    axs[0, 0].set_xlim([-3, 3])
    axs[0, 0].set_ylim([-3, 3])
    axs[0, 1].set_xlim([-3, 3])
    axs[0, 1].set_ylim([-3, 3])
    axs[1, 0].set_xlim([-3, 3])
    axs[1, 0].set_ylim([-3, 3])
    axs[1, 1].set_xlim([-3, 3])
    axs[1, 1].set_ylim([-3, 3])

    axs[0, 0].set_xlabel(f"$w_{index + 0}$")
    axs[0, 0].set_ylabel(f"$w_{index + 1}$")
    axs[0, 1].set_xlabel(f"$w_{index + 1}$")
    axs[0, 1].set_ylabel(f"$w_{index + 2}$")
    axs[1, 0].set_xlabel(f"$w_{index + 2}$")
    axs[1, 0].set_ylabel(f"$w_{index + 3}$")
    axs[1, 1].set_xlabel(f"$w_{index + 3}$")
    axs[1, 1].set_ylabel(f"$w_{index + 4}$")

    if name is not None:
        plt.savefig(name)
    plt.close()

    return fig
