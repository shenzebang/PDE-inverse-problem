import jax.numpy as jnp
import jax
from utils.common_utils import v_matmul
import warnings

class Potential(object):
    def gradient(self, x: jnp.ndarray):
        raise NotImplementedError


class QuadraticPotential(Potential):
    def __init__(self, mu: jnp.ndarray, cov: jnp.ndarray):
        assert mu.ndim == 1 and cov.ndim == 2 and cov.shape[0] == cov.shape[1] and cov.shape[0] == mu.shape[0]
        warnings.warn("cov is assumed to be positive definite!")
        self.dim = mu.shape[0]
        self.mu = mu
        self.cov = cov
        self.inv_cov = jnp.linalg.inv(self.cov)

    def gradient(self, x):
        if x.ndim == 1:
            return self.inv_cov @ (x - self.mu)
        else:
            return v_matmul(self.inv_cov, x - self.mu)


class VoidPotential(Potential):
    def gradient(self, x: jnp.ndarray):
        return jnp.zeros_like(x)


def gmm_V(x: jnp.ndarray, mus: jnp.ndarray, sigma: jnp.ndarray):
    # we use the broadcasting mechanism
    a = - jnp.sum((x-mus)**2, axis=(1,))/(2 * sigma ** 2)
    return - jax.scipy.special.logsumexp(a)

g_gmm_V = jax.grad(gmm_V) # by default, grad computes gradient w.r.t. the first input, i.e. argnums = 0

# def g_gmm_V(x, mus, sigma):
#     x = x - mus
#     a = jnp.exp(-jnp.sum(x**2, axis=1)/(2 * sigma**2))
#     normalized_a = a / jnp.sum(a)
#     return jnp.sum(x * normalized_a[:, None], axis=0)/sigma**2


vg_gmm_V = jax.vmap(g_gmm_V, in_axes=[0, None, None]) # only apply autobatching to the first input

class GMMPotential(Potential):
    def __init__(self, mus: jnp.ndarray, sigma: jnp.ndarray):
        # we assume that the Gaussian component has the same sigma for simplicity
        self.mus = mus
        self.sigma = sigma

    def value(self, x):
        return gmm_V(x, self.mus, self.sigma)

    def gradient(self, x):
        if len(x.shape) == 1:
            return g_gmm_V(x, self.mus, self.sigma)
        else:
            return vg_gmm_V(x, self.mus, self.sigma)

