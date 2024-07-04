import jax
import jax.numpy as jnp
import numpy as np
import optax


class Energy(object):
    def __init__(self):
        pass

    def __call__(self, z):
        raise NotImplementedError(str(type(self)))

    @staticmethod
    def mean():
        return None

    @staticmethod
    def std():
        return None

    def _vector_to_model(self, v):
        return v

    @staticmethod
    def statistics(z):
        return z

    def evaluate(self, z, path=None):
        raise NotImplementedError(str(type(self)))


class BayesianLogisticRegression(Energy):
    def __init__(self, data, labels, batch_size=None, mode="train", loc=0.0, scale=1.0):
        """
        Bayesian Logistic Regression model (assume Normal prior)
        :param data: data for Logistic Regression task
        :param labels: label for Logistic Regression task
        :param batch_size: batch size for Logistic Regression; setting it to None
        adds flexibility at the cost of speed.
        :param loc: mean of the Normal prior
        :param scale: std of the Normal prior
        """

        super(BayesianLogisticRegression, self).__init__()
        self.x_dim = data.shape[1]
        self.y_dim = labels.shape[1]
        self.dim = self.x_dim * self.y_dim + self.y_dim
        self.mu_prior = jnp.ones([self.dim]) * loc
        self.sig_prior = jnp.ones([self.dim]) * scale

        if mode == "train":
            num_samples = data.shape[0]
            self.data = jnp.array(data)[: num_samples // 10 * 9]
            self.labels = jnp.array(labels)[: num_samples // 10 * 9]

        elif mode == "test":
            num_samples = data.shape[0]
            self.data = jnp.array(data)[num_samples // 10 * 9 :]
            self.labels = jnp.array(labels)[num_samples // 10 * 9 :]

        else:
            self.data = jnp.array(data)
            self.labels = jnp.array(labels)

        self.z = jnp.zeros([batch_size, self.dim])

        if batch_size:
            self.data = jnp.tile(
                jnp.reshape(self.data, [1, -1, self.x_dim]),
                jnp.stack([batch_size, 1, 1]),
            )
            self.labels = jnp.tile(
                jnp.reshape(self.labels, [1, -1, self.y_dim]),
                jnp.stack([batch_size, 1, 1]),
            )
        else:
            self.data = jnp.tile(
                jnp.reshape(self.data, [1, -1, self.x_dim]),
                jnp.stack([jnp.shape(self.z)[0], 1, 1]),
            )
            self.labels = jnp.tile(
                jnp.reshape(self.labels, [1, -1, self.y_dim]),
                jnp.stack([jnp.shape(self.z)[0], 1, 1]),
            )

        self.create_fn()

    def create_fn(self):

        def energy_fn(v, x, y, mu_prior, sig_prior, x_dim, y_dim):
            w = v[:, :-y_dim]
            b = v[:, -y_dim:]
            w = jnp.reshape(w, [-1, x_dim, y_dim])
            b = jnp.reshape(b, [-1, 1, y_dim])
            logits = jnp.matmul(x, w) + b
            ll = optax.sigmoid_binary_cross_entropy(logits, y)
            ll = jnp.sum(ll, axis=[1, 2])
            pr = jnp.square((v - mu_prior) / sig_prior)
            pr = 0.5 * jnp.sum(pr, axis=1)
            return pr + ll

        self.energy_fn = jax.jit(energy_fn, static_argnames=["x_dim", "y_dim"])

        def potential_fn(p, inv_sigma):
            return 0.5 * p @ inv_sigma @ p.T

        self.potential_fn = jax.vmap(potential_fn, in_axes=(0, None))

        self.hamiltonian_fn = lambda v: self.energy_fn(
            v[:, : self.dim],
            self.data,
            self.labels,
            self.mu_prior,
            self.sig_prior,
            self.x_dim,
            self.y_dim,
        ) + self.potential_fn(v[:, self.dim :], jnp.eye(self.dim) * 1.0)

        # Gradient of the energy function

        def scalar_energy_fn(v, x, y, mu_prior, sig_prior, x_dim, y_dim):
            w = v[:-y_dim]
            b = v[-y_dim:]
            w = jnp.reshape(w, [x_dim, y_dim])
            b = jnp.reshape(b, [1, y_dim])
            logits = jnp.matmul(x, w) + b
            ll = optax.sigmoid_binary_cross_entropy(logits, y)
            ll = jnp.sum(ll, axis=0)
            pr = jnp.square((v - mu_prior) / sig_prior)
            pr = 0.5 * jnp.sum(pr, axis=0)
            return (pr + ll)[0]

        self.grad_energy_fn = jax.vmap(
            lambda v: jax.grad(scalar_energy_fn, argnums=0)(
                v,
                self.data[0],
                self.labels[0],
                self.mu_prior,
                self.sig_prior,
                self.x_dim,
                self.y_dim,
            )
        )

    def get_grad_energy_fn(self):
        return self.grad_energy_fn

    def __call__(self, v):
        return self.hamiltonian_fn(v)

    def sigmoid(self, v, x, y, x_dim, y_dim):
        w = v[:, :-y_dim]
        b = v[:, -y_dim:]
        w = jnp.reshape(w, [-1, x_dim])  # w.shape = [chain_length, x_dim] = [5000, 13]
        b = b[:, 0]  # b.shape = [chain_length] = [5000]
        logits = jnp.matmul(w, x) + b  # logits.shape = [chain_length] = [5000]
        ll = optax.sigmoid_binary_cross_entropy(
            logits, y
        )  # ll.shape = [chain_length] = [5000]
        return ll

    def new_instance(self, new_batch_size):
        return self.__class__(name=self.name, batch_size=new_batch_size, mode=self.mode)

    @staticmethod
    def mean():
        return None

    @staticmethod
    def std():
        return None
