import jax.numpy as jnp
from core.distribution import Gaussian, Uniform
import jax
from jax.experimental.ode import odeint
from api import ProblemInstance
from utils.common_utils import v_gaussian_score, v_gaussian_log_density
from core.potential import QuadraticPotential
from math import prod
import warnings
from flax import linen as nn

# To ensure that the coefficient of the Laplacian term in the FPE is 1, L should be 2


def initialize_configuration(domain_dim: int):
    _F = jax.random.normal(jax.random.PRNGKey(2217), (domain_dim, domain_dim + 1))
    tilde_F = (
        _F @ _F.transpose()
    )  # Symmetry is necessary, otherwise, it does not correspond to a gradient field.

    gamma_friction = 1.0

    tilde_L_scale = 2.0

    m_x_0_scale = 1.0
    P_x_0_scale = 1.0

    m_v_0_scale = 0.0
    P_v_0_scale = 1.0

    m_x_0 = jnp.zeros(domain_dim)
    # m_x_0 = jnp.ones(domain_dim) * m_x_0_scale
    m_v_0 = jnp.zeros(domain_dim) * m_v_0_scale

    m_0 = jnp.concatenate([m_x_0, m_v_0])

    P_x_0 = jnp.eye(domain_dim) * P_x_0_scale
    # P_x_0 = jnp.linalg.inv(tilde_F)
    P_v_0 = jnp.eye(domain_dim) * P_v_0_scale

    P_0 = jnp.block(
        [[P_x_0, jnp.eye(domain_dim) * 0.0], [jnp.eye(domain_dim) * 0.0, P_v_0]]
    )

    # tilde_F = jnp.eye(domain_dim)
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
    # print(state_T["m"][1:].shape, state_T["P"][1:].shape)
    # print(state_T["m"][-1].shape, state_T["P"][-1].shape)
    # return state_T["m"][1:], state_T["P"][1:]
    if jnp.size(t_space) == 2:
        return state_T["m"][-1], state_T["P"][-1]
    else:
        return state_T["m"][1:], state_T["P"][1:]


def get_mean_cov(t: jnp.ndarray, configuration):
    if jnp.size(t) == 1:
        # if t is a single time stamp, compute mean and cov only at t
        return OU_process(jnp.array([0.0, t]), configuration)
        # return Gaussian(mean, cov)
    else:
        # if t is a collection of time stamp, sort t and generate the means and covs accordingly
        assert t.ndim == 1  # ensure that this is a 1-D array
        # t = jnp.sort(t)
        warnings.warn("The user is responsible for ensuring t[0] == 0")
        return OU_process(t, configuration)


class KineticFokkerPlanck(ProblemInstance):
    def __init__(self, cfg, rng):
        super().__init__(cfg, rng)
        self.initial_configuration = initialize_configuration(
            cfg.pde_instance.domain_dim
        )
        self.get_mean_cov = lambda t: get_mean_cov(t, self.initial_configuration)
        self.distribution_initial = Gaussian(
            self.initial_configuration["m_0"], self.initial_configuration["P_0"]
        )
        self.distribution_initial_x = Gaussian(
            self.initial_configuration["m_x_0"], self.initial_configuration["P_x_0"]
        )
        self.distribution_terminal = Gaussian(
            *self.get_mean_cov(self.total_evolving_time)
        )

        if self.sample_mode == "offline":
            raise NotImplementedError
        # self.get_distribution(self.total_evolving_time)

    def V_true_fn(self, x: jnp.ndarray):
        _V_true_fn = lambda x: jnp.dot(x, self.initial_configuration["tilde_F"] @ x) / 2
        _V_true_fn_vmap_x = jax.vmap(_V_true_fn, in_axes=[0])
        if x.ndim == 1:
            return _V_true_fn(x)
        elif x.ndim == 2:
            return _V_true_fn_vmap_x(x)
        else:
            raise ValueError("x should be either 1D (unbatched) or 2D (batched) array.")

    def sample_ground_truth(self, rng, batch_size):
        if isinstance(batch_size, int):  # sample 100 random time
            sample_per_time = 100
            assert batch_size >= sample_per_time * 2
            n_random_time = batch_size // sample_per_time

            # sample a single data
            def _sample_ground_truth_fn(rng):
                rng_time, rng_x = jax.random.split(rng, 2)
                # sample time
                t = self.distribution_time.sample(1, rng_time)[0]
                # sample data
                x = Gaussian(*self.get_mean_cov(t)).sample(sample_per_time, rng_x)
                return x

            _sample_ground_truth_fn = jax.vmap(_sample_ground_truth_fn, in_axes=[0])
            samples = _sample_ground_truth_fn(jax.random.split(rng, n_random_time))

        else:  # grid [0, T] into n_time_stamps intervals
            rng_time_shift, rng = jax.random.split(rng)
            n_time_stamps = batch_size[0]
            sample_per_time = batch_size[1]
            random_time_shift = jax.random.uniform(
                rng_time_shift, [n_time_stamps + 1]
            ) * (self.total_evolving_time / n_time_stamps)
            time_stamps = (
                jnp.linspace(0, self.total_evolving_time, n_time_stamps + 1)
                + random_time_shift
            )  # the last time stamp is not to be included since it is out of bound
            time_stamps = jnp.concatenate(
                [jnp.zeros([1]), time_stamps[:-1]]
            )  # exclude the last time stamp
            rngs = jax.random.split(rng, n_time_stamps)
            means, covs = self.get_mean_cov(time_stamps)
            means = means.reshape(n_time_stamps, means.shape[-1])
            covs = covs.reshape(n_time_stamps, covs.shape[-1], covs.shape[-1])
            assert n_time_stamps == 1

            # TODO: debug when n_time_stamps > 1
            # means, covs = means[1:], covs[1:]
            @jax.vmap
            def _sample_ground_truth_fn(mean, cov, rng):
                return Gaussian(mean, cov).sample(sample_per_time, rng)

            samples = _sample_ground_truth_fn(means, covs, rngs).reshape(
                sample_per_time, n_time_stamps, -1
            )

        return samples.reshape(
            (prod(samples.shape[:2]), *samples.shape[2:])
        )  # combine the first two dimensions

    def get_time_sample_ground_truth(self, rng, batch_size):
        if isinstance(batch_size, int):  # sample 100 random time
            raise NotImplementedError
        else:  # grid [0, T] into n_time_stamps intervals
            rng_time_shift, rng = jax.random.split(rng)
            n_time_stamps = batch_size[0]
            random_time_shift = jax.random.uniform(
                rng_time_shift, [n_time_stamps + 1]
            ) * (self.total_evolving_time / n_time_stamps)
            time_stamps = (
                jnp.linspace(0, self.total_evolving_time, n_time_stamps + 1)
                + random_time_shift
            )  # the last time stamp is not to be included since it is out of bound
            time_stamps = time_stamps[:-1]  # exclude the last time stamp

        return time_stamps

    def create_parametric_model(self):
        dim = self.dim

        class V_parametric(nn.Module):
            def setup(self):
                # Create a one-layer MLP for the quadratic potential
                self.tilde_F = nn.Dense(dim)

            def __call__(self, y_input: jnp.ndarray):
                return jnp.sum(y_input * self.tilde_F(y_input), axis=-1)[None]

        return V_parametric()
