import jax
import jax.numpy as jnp
from jax import grad


statistics_mog2 = {'mu': [0., 0.],
              'sigma': [0.7, 5.05]
              }

def normal(x, mu, inv_cov):
    return jnp.exp(-0.5 * jnp.dot(jnp.dot((x - mu).T, inv_cov), (x - mu)))
    
normal = jax.vmap(normal, in_axes=(0, None, None))

def mog2(x, mu1=jnp.array([5.0, 0.0]), mu2=jnp.array([-5.0, 0.0]), inv_cov=jnp.eye(2) * 2):
    return -jnp.log(0.5 * normal(x, mu1, inv_cov) + 0.5 * normal(x, mu2, inv_cov))

def hamiltonian_mog2(
    x,
    mu1=jnp.array([5.0, 0.0]),
    mu2=jnp.array([-5.0, 0.0]),
    inv_cov=jnp.eye(2) * 2,
    inv_cov_p=jnp.eye(2),
):
    d = x.shape[1]
    return mog2(x[:, : d // 2], mu1, mu2, inv_cov) - jnp.log(normal(x[:, d // 2 :], jnp.zeros(d // 2), inv_cov_p))

def nv_normal(x, mu, inv_cov):
    return jnp.exp(-0.5 * jnp.dot(jnp.dot((x - mu).T, inv_cov), (x - mu)))

def nv_mog2(x, mu1=jnp.array([5.0, 0.0]), mu2=jnp.array([-5.0, 0.0]), inv_cov=jnp.eye(2) * 2):
    return -jnp.log(0.5 * nv_normal(x, mu1, inv_cov) + 0.5 * nv_normal(x, mu2, inv_cov))

grad_mog2 = jax.vmap(grad(nv_mog2))
