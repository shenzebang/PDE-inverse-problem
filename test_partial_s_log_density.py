import jax.numpy as jnp
import jax
from jax.experimental.ode import odeint
from core.distribution import Gaussian
from api import ProblemInstance
import warnings


def initialize_configuration(domain_dim: int):
    gamma_friction = 0.1

    tilde_F_scale = 1.0
    tilde_L_scale = 2.0

    m_x_0_scale = 1.0
    P_x_0_scale = 1.0

    m_v_0_scale = 0.0
    P_v_0_scale = 0.1

    # m_x_0 = jnp.ones(domain_dim) * m_x_0_scale
    m_x_0 = jnp.zeros(domain_dim) * m_x_0_scale
    m_v_0 = jnp.zeros(domain_dim) * m_v_0_scale

    m_0 = jnp.concatenate([m_x_0, m_v_0])

    P_x_0 = jnp.eye(domain_dim) * P_x_0_scale
    P_v_0 = jnp.eye(domain_dim) * P_v_0_scale

    P_0 = jnp.block(
        [[P_x_0, jnp.eye(domain_dim) * 0.0], [jnp.eye(domain_dim) * 0.0, P_v_0]]
    )

    _F = jax.random.normal(jax.random.PRNGKey(2217), (domain_dim, domain_dim + 1))
    tilde_F = (
        _F @ _F.transpose() * tilde_F_scale
    )  # Symmetry is necessary, otherwise, it does not correspond to a gradient field.
    F = jnp.block(
        [
            [jnp.eye(domain_dim) * 0.0, jnp.eye(domain_dim)],
            [-tilde_F, -jnp.eye(domain_dim) * gamma_friction],
        ]
    )

    tilde_L = jnp.eye(domain_dim) * tilde_L_scale
    L = jnp.block(
        [
            [jnp.eye(domain_dim) * 0.0, jnp.eye(domain_dim) * 0],
            [jnp.eye(domain_dim) * 0.0, tilde_L],
        ]
    )

    return {
        "gamma_friction": gamma_friction,
        "tilde_F": tilde_F,
        "F": F,
        "L": L,
        "m_0": m_0,
        "P_0": P_0,
        "m_x_0": m_x_0,
        "P_x_0": P_x_0,
    }


def OU_process(t_space: jnp.ndarray, configuration):
    assert jnp.size(t_space) >= 2
    # compute the mean and variance according to the ODE
    state_0 = {"m": configuration["m_0"], "P": configuration["P_0"]}

    def ode_func(states, t):
        return {
            "m": configuration["F"] @ states["m"],
            "P": configuration["F"] @ states["P"]
            + states["P"] @ configuration["F"].transpose()
            + configuration["L"],
        }

    state_T = odeint(ode_func, state_0, t_space)
    if jnp.size(t_space) == 2:
        return state_T["m"][-1], state_T["P"][-1]
    else:
        return state_T["m"][1:], state_T["P"][1:]


def get_mean_cov(t: jnp.ndarray, configuration):
    if jnp.size(t) == 1:
        # if t is a single time stamp, compute mean and cov only at t
        return OU_process(jnp.array([0.0, t]), configuration)
    # else:
    #     # if t is a collection of time stamp, sort t and generate the means and covs accordingly
    #     assert t.ndim == 1  # ensure that this is a 1-D array
    #     t = jnp.sort(t)
    #     warnings.warn("The user is responsible for ensuring t[0] == 0")
    #     return OU_process(t, configuration)


class KineticMcKeanVlasov(ProblemInstance):
    def __init__(self, cfg, rng):
        super().__init__(cfg, rng)
        self.initial_configuration = initialize_configuration(
            cfg.pde_instance.domain_dim
        )
        self.get_mean_cov = lambda t: get_mean_cov(t, self.initial_configuration)

    def partial_s_log_density_fn(self, s: jnp.ndarray, x: jnp.ndarray):
        def _partial_s_log_density_fn(s: jnp.ndarray, x: jnp.ndarray):
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
            # term1 = 0
            term2 = -0.5 * jnp.trace(dP_11ds @ cov11_inv)
            # term2 = 0
            term3 = -0.5 * jnp.dot(mean1 - x, dP_11ds_inv @ (mean1 - x))
            # term3 = 0
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

    def log_density_fn(self, s: jnp.ndarray, x: jnp.ndarray):
        def _log_density_fn(s: jnp.ndarray, x: jnp.ndarray):
            mean, cov = self.get_mean_cov(s)
            mean1 = mean[: self.dim]
            cov11 = cov[: self.dim, : self.dim]
            cov11_inv = jnp.linalg.inv(cov11)
            term1 = -0.5 * jnp.dot(mean1 - x, cov11_inv @ (mean1 - x))
            # term1 = 0
            term2 = -0.5 * jnp.log(jnp.linalg.det(2 * jnp.pi * cov11))
            # term2 = 0
            return term1 + term2

        _log_density_fn_vmap = jax.vmap(_log_density_fn, in_axes=(0, None))
        if s.ndim == 0 and x.ndim == 1:
            return _log_density_fn(s, x)
        elif s.ndim == 1 and x.ndim == 1:
            return _log_density_fn_vmap(s, x)
        elif s.ndim == 0 and x.ndim == 2:
            return jax.vmap(_log_density_fn, in_axes=(None, 0))(s, x)
        elif s.ndim == 1 and x.ndim == 2:
            return jax.vmap(_log_density_fn_vmap, in_axes=(None, 0))(s, x)
        else:
            raise ValueError("Shapes of s and x are not supported.")

    def partial_s2_log_density_fn(self, s: jnp.ndarray, x: jnp.ndarray):
        def _partial_s2_log_density_fn(s: jnp.ndarray, x: jnp.ndarray):
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


# Example test case
if __name__ == "__main__":

    class Config:
        class PDEInstance:
            domain_dim = 10
            name = "KineticMcKeanVlasov"
            diffusion_coefficient = 1.0
            total_evolving_time = 1.0

        pde_instance = PDEInstance()
        sample_mode = "online"

    cfg = Config()
    rng = jax.random.PRNGKey(0)
    problem_instance = KineticMcKeanVlasov(cfg, rng)

    # Sample s and x randomly
    num_samples = 3
    # s_samples = jax.random.uniform(rng, (num_samples,))
    s_samples = jnp.zeros([]) + 0.1
    x_samples = jax.random.uniform(rng, [num_samples, cfg.pde_instance.domain_dim])
    # x_samples = jax.random.uniform(rng, (num_samples, cfg.pde_instance.domain_dim))

    # Testing first order derivative of log_density
    # Calculate the partial_s_log_density for all samples
    partial_s_log_density_results = problem_instance.partial_s_log_density_fn(
        s_samples, x_samples
    )
    # Calculate the numerical derivative of log_density for all samples
    delta_s = 1e-4
    numerical_derivative_results = (
        problem_instance.log_density_fn(s_samples + delta_s, x_samples)
        - problem_instance.log_density_fn(s_samples - delta_s, x_samples)
    ) / (delta_s * 2)
    # Calculate relative RMSE
    rmse = jnp.sqrt(
        jnp.mean(
            (
                (partial_s_log_density_results - numerical_derivative_results)
                / numerical_derivative_results
            )
            ** 2
        )
    )
    print("Relative RMSE:", rmse)

    # =================================================================================================
    # Testing second order derivative of log_density
    # Calculate the partial_s2_log_density for all samples
    partial_s2_log_density_results = problem_instance.partial_s2_log_density_fn(
        s_samples, x_samples
    )

    # Calculate the numerical derivative of log_density for all samples
    delta_s = 1e-3
    numerical_derivative_results = (
        problem_instance.partial_s_log_density_fn(s_samples + delta_s, x_samples)
        - problem_instance.partial_s_log_density_fn(s_samples - delta_s, x_samples)
    ) / (delta_s * 2)

    # Calculate relative RMSE
    rmse = jnp.sqrt(
        jnp.mean(
            (
                (partial_s2_log_density_results - numerical_derivative_results)
                / numerical_derivative_results
            )
            ** 2
        )
    )
    print("Relative RMSE:", rmse)
    # print(s_samples, x_samples)
    # print(problem_instance.initial_configuration["F"])
    # print(partial_s_log_density_results, numerical_derivative_results)
    # print(
    #     (partial_s_log_density_results - numerical_derivative_results)
    #     / numerical_derivative_results
    # )

    # x = jnp.array([1.0, 2.0, 3.0, 4.0])
    # result = problem_instance.partial_s_log_density_fn(s, x)
    # print(result)
    # delta_s = 1e-4
    # result = (
    #     (
    #         problem_instance.log_density_fn(s + delta_s, x)
    #         - problem_instance.log_density_fn(s - delta_s, x)
    #     )
    #     / delta_s
    #     / 2
    # )
    # print(result)
