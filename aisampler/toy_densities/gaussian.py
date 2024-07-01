import jax
import jax.numpy as jnp

def normal(x, mu, cov):
    d = x.shape[0]
    return jnp.exp(-0.5 * jnp.dot(jnp.dot((x - mu).T, jnp.linalg.inv(cov)), (x - mu))) * (1 / jnp.sqrt( 2 * (jnp.pi**d) * jnp.linalg.det(cov)))

normal = jax.vmap(normal, in_axes=(0, None, None))

def N(x, mu=jnp.array([0., 0.]), cov=jnp.eye(2)*0.5):
    return jnp.log(normal(x, mu, cov))

def hamiltonian_N(x, cov_p=jnp.eye(2)*0.5):
    d = x.shape[1]
    return N(x[:, : d // 2]) + jnp.log(normal(x[:, d // 2 :], jnp.zeros(2), cov_p))