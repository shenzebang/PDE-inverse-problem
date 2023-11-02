import jax
import jax.numpy as jnp
import math




def _divergence_fn(f, _x, _v):
    # Hutchinson’s Estimator
    # computes the divergence of net at x with random vector v
    _, u = jax.jvp(f, (_x,), (_v,))
    # print(u.shape, _x.shape, _v.shape)
    return jnp.sum(u * _v)


# f_list = [lambda x: f(x)[i]]

def _divergence_bf_fn(f, _x):
    # brute-force implementation of the divergence operator
    # _x should be a d-dimensional vector
    jacobian = jax.jacfwd(f)
    a = jacobian(_x)
    return jnp.sum(jnp.diag(a))



batch_div_bf_fn = jax.vmap(_divergence_bf_fn, in_axes=[None, 0])

batch_div_fn = jax.vmap(_divergence_fn, in_axes=[None, None, 0])


def divergence_fn(f, _x: jnp.ndarray, _v=None):
    if _v is None:
        if _x.ndim == 1:
            return _divergence_bf_fn(f, _x)
        return batch_div_bf_fn(f, _x)
    else:
        return batch_div_fn(f, _x, _v).mean(axis=0)


def _gaussian_score(x, cov, mu): # return the score for a given Gaussian(mu, Sigma) at x
    return jax.numpy.linalg.inv(cov) @ (mu - x)

# return the score for a given Gaussians(Sigma, mu) at [x1, ..., xN]
v_gaussian_score = jax.vmap(_gaussian_score, in_axes=[0, None, None])

def _gaussian_log_density(x, cov, mu):
    log_det = jnp.log(jax.numpy.linalg.det(cov * 2 * jnp.pi))
    inv_cov = jax.numpy.linalg.inv(cov)
    quad = jnp.dot(x - mu, inv_cov @ (x - mu))
    return - .5 * (log_det + quad)

v_gaussian_log_density = jax.vmap(_gaussian_log_density, in_axes=[0, None, None])

v_matmul = jax.vmap(jnp.matmul, in_axes=(None, 0))


def volume_nd_ball(d: int):
    k = d // 2
    if d % 2 == 0:
        return jnp.pi ** k / math.factorial(k)
    else:
        return 2 * math.factorial(k) * ((4 * jnp.pi) ** k) / math.factorial(d)

def compute_pytree_norm(pytree):
    pytree_norm = jnp.sqrt(sum(jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(pytree)))
    return pytree_norm