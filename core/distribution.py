from abc import ABC

import jax.numpy as jnp
import jax.random
import jax.random as random
from utils.common_utils import v_matmul
from typing import List
import warnings


class Distribution(ABC):
    def sample(self, batch_size: int, key):
        raise NotImplementedError

    def score(self, x: jnp.ndarray):
        raise NotImplementedError

    def logdensity(self, x: jnp.ndarray):
        raise NotImplementedError
    
    def density(self, x: jnp.ndarray):
        raise NotImplementedError


class DistributionKinetic(Distribution):
    def __init__(self, distribution_x: Distribution, distribution_v: Distribution):
        warnings.warn("Currently DistributionKinetic only supports when x and v are independent")
        self.distribution_x = distribution_x
        self.distribution_v = distribution_v

    def sample(self, batch_size: int, key):
        key_x, key_v = jax.random.split(key, 2)
        x = self.distribution_x.sample(batch_size, key_x)
        v = self.distribution_v.sample(batch_size, key_v)
        z = jnp.concatenate([x, v], axis=-1)
        return z

    def score(self, z: jnp.ndarray):
        x, v = jnp.split(z, indices_or_sections=2, axis=-1)
        score_x = self.distribution_x.score(x)
        score_v = self.distribution_v.score(v)
        score_z = jnp.concatenate([score_x, score_v], axis=-1)
        return score_z

    def logdensity(self, z: jnp.ndarray):
        x, v = jnp.split(z, indices_or_sections=2, axis=-1)
        logdensity_x = self.distribution_x.logdensity(x)
        logdensity_v = self.distribution_v.logdensity(v)
        logdensity_z = logdensity_x + logdensity_v
        return logdensity_z

class Gaussian(Distribution):
    def __init__(self, mu: jnp.ndarray, cov: jnp.ndarray):
        assert mu.ndim == 1 and cov.ndim == 2 and cov.shape[0] == cov.shape[1] and cov.shape[0] == mu.shape[0]
        warnings.warn("cov is assumed to be positive definite!")
        self.dim = mu.shape[0]
        self.mu = mu
        self.cov = cov
        U, S, _ = jnp.linalg.svd(cov)
        self.inv_cov = jnp.linalg.inv(self.cov)
        self.log_det = jnp.log(jnp.linalg.det(self.cov * 2 * jnp.pi))
        self.cov_half = U @ jnp.diag(jnp.sqrt(S)) @ jnp.transpose(U)

    def sample(self, batch_size: int, key):
        return v_matmul(self.cov_half, random.normal(key, (batch_size, self.dim))) + self.mu

    def score(self, x: jnp.ndarray):
        if x.ndim == 1:
            return jnp.matmul(self.inv_cov, self.mu - x)
        else:
            return v_matmul(self.inv_cov, self.mu - x)

    def logdensity(self, x: jnp.ndarray):
        if x.ndim == 1:  # no batching
            quad = jnp.dot(x - self.mu, self.inv_cov @ (x - self.mu))
        elif x.ndim == 2:  # the first dimension is batch
            offset = x - self.mu  # broadcasting
            quad = jnp.sum(offset * v_matmul(self.inv_cov, offset), axis=(-1,))
        else:
            raise NotImplementedError
        return - .5 * (self.log_det + quad)

    def density(self, x: jnp.ndarray):
        return jnp.exp(self.logdensity(x))

class Uniform_over_3d_Ball(Distribution):
    def __init__(self, r):
        self.r = r

    def sample(self, batch_size: int, key):
        return jax.random.ball(key, d=3, p=2, shape=[batch_size]) * self.r

    def score(self, x: jnp.ndarray):
        return jnp.zeros_like(x)


class GaussianMixture(Distribution):
    def __init__(self, mus: List[jnp.ndarray], sigmas: List[jnp.ndarray]):
        # we assume uniform weight among the Gaussians
        self.n_Gaussians = len(mus)
        assert self.n_Gaussians == len(sigmas)
        # assert self.n_Gaussians == weights.shape[0]
        # assert all(weights > 0)

        # self.weights = weights / jnp.sum(weights)
        self.dim = mus[0].shape[0]

        self.mus = mus
        self.sigmas = sigmas
        self.covs, self.inv_covs, self.dets = [], [], []

        for sigma in sigmas:
            if sigma.ndim == 2:
                assert sigma.shape[0] == sigma.shape[1]  # make sure sigma is a square matrix
                # if sigma.shape[0] != 1, the covariance matrix is sigma.transpose * sigma
                cov = jnp.matmul(sigma, jnp.transpose(sigma))
                inv_cov = jnp.linalg.inv(cov)
                det = jnp.linalg.det(cov)
            else:
                # sigma is a scalar
                cov = sigma ** 2
                inv_cov = 1. / cov
                det = sigma ** (2 * self.dim)

            self.covs.append(cov)
            self.inv_covs.append(inv_cov)
            self.dets.append(det)

        self.mus = jnp.stack(self.mus)
        self.covs = jnp.stack(self.covs)
        self.inv_covs = jnp.stack(self.inv_covs)
        self.dets = jnp.stack(self.dets)

    def sample(self, batch_size: int, key):
        n_sample_per_center = []
        remainder = batch_size % self.n_Gaussians
        for i in range(self.n_Gaussians):
            n_sample_i = batch_size // self.n_Gaussians
            if remainder != 0:
                n_sample_i += 1
                remainder -= 1
            n_sample_per_center.append(n_sample_i)

        samples = []
        keys = jax.random.split(key, self.n_Gaussians)
        for i, (n_sample_i, _key) in enumerate(zip(n_sample_per_center, keys)):
            mu, sigma = self.mus[i, :], self.sigmas[i]
            if sigma.ndim == 1:
                samples.append(sigma * random.normal(_key, (n_sample_i, self.dim)) + mu)
            else:
                samples.append(v_matmul(sigma, random.normal(_key, (n_sample_i, self.dim))) + mu)

        return jnp.concatenate(samples, axis=0)

    def logdensity(self, xs: jnp.ndarray):
        return v_logdensity_gmm(xs, self.mus, self.inv_covs, self.dets)

    def score(self, xs: jnp.ndarray):
        return v_score_gmm(xs, self.mus, self.inv_covs, self.dets)


class Uniform(Distribution):
    def __init__(self, mins: jnp.ndarray, maxs: jnp.ndarray):
        if mins.ndim != maxs.ndim:
            raise ValueError("mins and maxs should be arrays of the same size") 
        if mins.ndim == 1:
            if len(mins) != len(maxs):
                raise ValueError("mins and maxs should be of the same dimension") 
        elif mins.ndim == 0:
            pass
        else:
            raise ValueError("mins and maxs should be either 0D or 1D arrays")    
        
        self.dim =  mins.shape[0] if mins.ndim == 1 else 0
        self.mins = mins
        self.maxs = maxs

    def sample(self, batch_size: int, key):
        shape = [batch_size, self.dim] if self.dim !=0 else [batch_size]
        return jax.random.uniform(key=key, shape=shape, minval=self.mins, maxval=self.maxs)

    def score(self, x: jnp.ndarray):
        pass

    def logdensity(self, x: jnp.ndarray):
        pass


class UniformMixture(Distribution):
    def __init__(self, uniforms: List[Uniform]):
        self.uniforms = uniforms
        self.n_uniforms = len(uniforms)

    def sample(self, batch_size: int, key):
        if batch_size % self.n_uniforms != 0:
            raise ValueError(f"batch_size should be a multiple of n_uniforms {self.n_uniforms}!")

        _n = batch_size // self.n_uniforms
        _samples = []
        _keys = jax.random.split(key, self.n_uniforms)
        for _key, uniform in zip(_keys, self.uniforms):
            _samples.append(uniform.sample(_n, _key))
        return jnp.concatenate(_samples)


def get_uniforms_over_box_boundary(mins: jnp.ndarray, maxs: jnp.ndarray):
    if not (mins.ndim == 1 and maxs.ndim == 1):
        raise ValueError("mins and maxs should be 1-d array")
    if mins.shape[0] != maxs.shape[0]:
        raise ValueError("mins and maxs should have the same length")
    dim = mins.shape[0]
    uniforms = []
    for i in range(dim):
        basis_i = [0.] * dim
        basis_i[i] = 1.
        basis_i = jnp.array(basis_i)
        min_i, max_i = mins[i], maxs[i]
        _mins_i_min = mins
        _maxs_i_min = maxs + (-max_i + min_i) * basis_i
        uniforms.append(Uniform(_mins_i_min, _maxs_i_min))
        _mins_i_max = mins + (-min_i + max_i) * basis_i
        _maxs_i_max = maxs
        uniforms.append(Uniform(_mins_i_max, _maxs_i_max))

    return uniforms


def _density_gaussian(x, mu, inv_cov, det):
    # computes the density in a single Gaussian of a single point
    a = x - mu
    dim = x.shape[0]
    if inv_cov.ndim == 1:
        return jnp.squeeze(jnp.exp(- .5 * jnp.dot(a, a) * inv_cov) / jnp.sqrt((2 * jnp.pi) ** dim * det))
    else:
        return jnp.exp(- .5 * jnp.dot(a, jnp.matmul(inv_cov, a))) / jnp.sqrt((2 * jnp.pi) ** dim * det)


v_density_gaussian = jax.vmap(_density_gaussian, in_axes=[None, 0, 0, 0])


# computes the density in several Gaussians of a single point


def _logdensity_gmm(x, mus, inv_covs, dets):
    # computes log densities of gmm of multiple points
    densities = v_density_gaussian(x, mus, inv_covs, dets)
    # densities : (self.n_Gaussians)
    return jnp.log(jnp.mean(densities, axis=0))


v_logdensity_gmm = jax.vmap(_logdensity_gmm, in_axes=[0, None, None, None])
# computes log densities of gmm of multiple points

_score_gmm = jax.grad(_logdensity_gmm)
# compute the gradient w.r.t. x

v_score_gmm = jax.vmap(_score_gmm, in_axes=[0, None, None, None])
