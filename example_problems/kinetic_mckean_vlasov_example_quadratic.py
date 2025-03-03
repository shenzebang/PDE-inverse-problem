import jax.numpy as jnp
from core.distribution import Gaussian, Uniform
import jax
from jax.experimental.ode import odeint
from api import ProblemInstance
from example_problems.kinetic_fokker_planck_example_OU import KineticFokkerPlanck
from utils.common_utils import v_gaussian_score, v_gaussian_log_density
from core.potential import QuadraticPotential
from math import prod
import warnings
from flax import linen as nn


# Since the interacting potential is quadratic, the (Phi \ast rho_t) is quadratic w.r.t. x.
# We can simply use the implementation in KineticFokkerPlanck.
# The only thing we need to do is to implement the the first- and second-order time derivatives of the log density function.
class KineticMcKeanVlasov(KineticFokkerPlanck):
    def partial_s_log_density_fn(self, s: jnp.ndarray, x: jnp.ndarray):
        """
        Compute the partial derivative of the log density function with respect to time `s`.

        This function computes the partial derivative of the log density function with respect to time `s`
        for a given state `x`. The function supports both scalar and vector inputs for `s` and `x`.

        Parameters:
        -----------
        s : jnp.ndarray
            The time variable. Can be a scalar (0-dimensional array) or a vector (1-dimensional array).
        x : jnp.ndarray
            The state variable. Can be a vector (1-dimensional array) or a matrix (2-dimensional array).

        Returns:
        --------
        jnp.ndarray
            The partial derivative of the log density function with respect to `s`. The shape of the output
            depends on the shapes of `s` and `x`.

        Raises:
        -------
        ValueError
            If the shapes of `s` and `x` are not supported.

        Notes:
        ------
        - If `s` is a scalar and `x` is a vector, the function returns a scalar.
        - If `s` is a vector and `x` is a vector, the function returns a vector.
        - If `s` is a scalar and `x` is a matrix, the function returns a vector.
        - If `s` is a vector and `x` is a matrix, the function returns a matrix.
        """

        def _partial_s_log_density_fn(s: jnp.ndarray, x: jnp.ndarray):
            assert jnp.size(s) == 1
            mean, cov = self.get_mean_cov(s)
            mean1 = mean[: self.dim]
            cov11 = cov[: self.dim, : self.dim]
            cov11_inv = jnp.linalg.inv(cov11)
            dmds = self.initial_configuration["F"] @ mean
            dm_1ds = dmds[: self.dim]
            dPds = (
                self.initial_configuration["F"] @ cov
                + cov @ self.initial_configuration["F"].transpose()
                + self.initial_configuration["L"]
            )
            dP_11ds = dPds[: self.dim, : self.dim]
            dP_11ds_inv = -cov11_inv @ dP_11ds @ cov11_inv
            term1 = -dm_1ds @ cov11_inv @ (mean1 - x)
            term2 = -0.5 * jnp.trace(dP_11ds @ cov11_inv)
            term3 = -0.5 * jnp.dot(mean1 - x, dP_11ds_inv @ (mean1 - x))
            return term1 + term2 + term3

        _partial_s_log_density_fn_vmap = jax.vmap(
            _partial_s_log_density_fn, in_axes=(0, None)
        )
        if s.ndim == 0 and x.ndim == 1:
            return _partial_s_log_density_fn(s, x)
        elif s.ndim == 1 and x.ndim == 1:
            return _partial_s_log_density_fn_vmap(s, x)
        elif s.ndim == 0 and x.ndim == 2:
            return jax.vmap(_partial_s_log_density_fn, in_axes=(None, 0))(s, x)
        elif s.ndim == 1 and x.ndim == 2:
            return jax.vmap(_partial_s_log_density_fn_vmap, in_axes=(None, 0))(s, x)
        else:
            raise ValueError("Shapes of s and x are not supported.")

    def partial_s2_log_density_fn(self, s: jnp.ndarray, x: jnp.ndarray):
        """
        Compute the second order partial derivative of the log density function with respect to s.

        This function computes the second order partial derivative of the log density function
        with respect to the time variable s, given the current state x. The computation involves
        the mean and covariance of the system at time s, as well as their first and second order
        derivatives with respect to s.

        Parameters:
        -----------
        s : jnp.ndarray
            The time variable(s). Can be a scalar or a 1-dimensional array.
        x : jnp.ndarray
            The state variable(s). Can be a 1-dimensional or 2-dimensional array.

        Returns:
        --------
        jnp.ndarray
            The second order partial derivative of the log density function with respect to s.
            The shape of the output depends on the shapes of the input variables s and x.

        Raises:
        -------
        ValueError
            If the shapes of s and x are not supported.

        Notes:
        ------
        - If s is a scalar and x is a 1-dimensional array, the function returns a scalar.
        - If s is a 1-dimensional array and x is a 1-dimensional array, the function returns a 1-dimensional array.
        - If s is a scalar and x is a 2-dimensional array, the function returns a 1-dimensional array.
        - If s is a 1-dimensional array and x is a 2-dimensional array, the function returns a 2-dimensional array.
        """

        def _partial_s2_log_density_fn(s: jnp.ndarray, x: jnp.ndarray):
            assert jnp.size(s) == 1
            # Get the mean and covariance at time s
            mean, cov = self.get_mean_cov(s)

            # Extract the mean and covariance for the position variables
            mean1 = mean[: self.dim]
            cov11 = cov[: self.dim, : self.dim]

            # Compute the inverse of the covariance matrix
            cov11_inv = jnp.linalg.inv(cov11)

            # Compute the derivative of the mean with respect to s
            dmds = self.initial_configuration["F"] @ mean
            dm_1ds = dmds[: self.dim]
            # Compute the second order derivative of the mean with respect to s
            d2mds2 = self.initial_configuration["F"] @ dmds
            d2m_1ds2 = d2mds2[: self.dim]

            # Compute the derivative of the covariance matrix with respect to s
            dPds = (
                self.initial_configuration["F"] @ cov
                + cov @ self.initial_configuration["F"].transpose()
                + self.initial_configuration["L"]
            )
            dP_11ds = dPds[: self.dim, : self.dim]
            # Compute the second order derivative of the covariance matrix with respect to s
            d2Pds2 = (
                self.initial_configuration["F"] @ dPds
                + dPds @ self.initial_configuration["F"].transpose()
            )
            d2P_11ds2 = d2Pds2[: self.dim, : self.dim]

            # Compute the derivative of the inverse of the covariance matrix
            dinv_P_11ds = -cov11_inv @ dP_11ds @ cov11_inv
            dinv_P_11_2ds2 = (
                -cov11_inv @ d2P_11ds2 @ cov11_inv
                + cov11_inv @ dP_11ds @ cov11_inv @ dP_11ds @ cov11_inv * 2
            )

            # Compute the terms of the second order partial derivative of the log density
            term1 = (
                -d2m_1ds2 @ cov11_inv @ (mean1 - x)
                - dm_1ds @ dinv_P_11ds @ (mean1 - x)
                - dm_1ds @ cov11_inv @ dm_1ds
            )
            # term1 = 0
            term2 = (
                -0.5 * (x - mean1) @ dinv_P_11_2ds2 @ (x - mean1)
                - (mean1 - x) @ dinv_P_11ds @ dm_1ds
            )
            # term2 = 0
            term3 = 0.5 * jnp.trace(
                cov11_inv @ dP_11ds @ cov11_inv @ dP_11ds
            ) - 0.5 * jnp.trace(cov11_inv @ d2P_11ds2)

            # Return the sum of the terms
            return term1 + term2 + term3

        _partial_s2_log_density_fn_vmap = jax.vmap(
            _partial_s2_log_density_fn, in_axes=(0, None)
        )
        if s.ndim == 0 and x.ndim == 1:
            return _partial_s2_log_density_fn(s, x)
        elif s.ndim == 1 and x.ndim == 1:
            return _partial_s2_log_density_fn_vmap(s, x)
        elif s.ndim == 0 and x.ndim == 2:
            return jax.vmap(_partial_s2_log_density_fn, in_axes=(None, 0))(s, x)
        elif s.ndim == 1 and x.ndim == 2:
            return jax.vmap(_partial_s2_log_density_fn_vmap, in_axes=(None, 0))(s, x)
        else:
            raise ValueError("Shapes of s and x are not supported.")

    def Phi_true_fn(self, x: jnp.ndarray):
        _Phi_true_fn = (
            lambda x: jnp.dot(x, self.initial_configuration["tilde_F"] @ x) / 2
        )
        _Phi_true_fn_vmap_x = jax.vmap(_Phi_true_fn, in_axes=[0])
        if x.ndim == 1:
            return _Phi_true_fn(x)
        elif x.ndim == 2:
            return _Phi_true_fn_vmap_x(x)
        else:
            raise ValueError("x should be either 1D (unbatched) or 2D (batched) array.")

    def create_parametric_model(self):
        dim = self.dim

        class Phi_parametric(nn.Module):
            def setup(self):
                # Create a one-layer MLP for the quadratic interaction
                self.tilde_F = nn.Dense(dim)

            def __call__(self, y_input: jnp.ndarray):
                return jnp.sum(y_input * self.tilde_F(y_input), axis=-1)[None]

        return Phi_parametric()
