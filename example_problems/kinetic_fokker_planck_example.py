import jax.numpy as jnp
from core.distribution import Gaussian, Uniform
import jax
from jax.experimental.ode import odeint
from api import ProblemInstance
from utils.common_utils import v_gaussian_score, v_gaussian_log_density
from core.potential import QuadraticPotential
from math import prod
# To ensure that the coefficient of the Laplacian term in the FPE is 1, L should be sqrt{2}


def initialize_configuration(domain_dim: int):
    tilde_F_scale = 1.
    tilde_L_scale = 2.

    m_x_0_scale = 1.
    P_x_0_scale = 1.

    m_v_0_scale = 0.
    P_v_0_scale = .1

    m_x_0 = jnp.ones(domain_dim) * m_x_0_scale
    m_v_0 = jnp.zeros(domain_dim) * m_v_0_scale

    m_0 = jnp.concatenate([m_x_0, m_v_0])

    P_x_0 = jnp.eye(domain_dim) * P_x_0_scale
    P_v_0 = jnp.eye(domain_dim) * P_v_0_scale

    P_0 = jnp.block([
        [P_x_0, jnp.eye(domain_dim) * 0.],
        [jnp.eye(domain_dim) * 0., P_v_0]
    ])
    
    _F = jax.random.normal(jax.random.PRNGKey(2217), (domain_dim, domain_dim + 1))
    tilde_F = _F @ _F.transpose() * tilde_F_scale # PD is not necessary
    F = jnp.block([
        [jnp.eye(domain_dim) * 0., jnp.eye(domain_dim)],
        [-tilde_F,            jnp.eye(domain_dim) * 0.]
    ])

    tilde_L = jnp.eye(domain_dim) * tilde_L_scale
    L = jnp.block([
        [jnp.eye(domain_dim) * 0., jnp.eye(domain_dim) * 0],
        [jnp.eye(domain_dim) * 0.,                 tilde_L],
    ])
    
    return {
        "tilde_F": tilde_F,
        "F": F,
        "L": L,
        "m_0": m_0,
        "P_0": P_0,
    }

def OU_process(t_space: jnp.ndarray, configuration): 
    assert jnp.size(t_space) == 2
    # compute the mean and variance according to the ODE
    state_0 = {"m": configuration["m_0"], "P": configuration["P_0"]}
    def ode_func(states, t):
        return {
            "m": configuration["F"] @ states["m"],
            "P": configuration["F"] @ states["P"] + states["P"] @ configuration["F"].transpose() + configuration["L"],
        }
    state_T = odeint(ode_func, state_0, t_space)
    return state_T["m"][-1], state_T["P"][-1]

def get_distribution(t: jnp.ndarray, configuration):
    mean, cov = OU_process(jnp.array([0., t]), configuration)
    return Gaussian(mean, cov)

class KineticFokkerPlanck(ProblemInstance):
    def __init__(self, cfg, rng):
        super().__init__(cfg, rng)
        self.initial_configuration = initialize_configuration(cfg.pde_instance.domain_dim)
        self.get_distribution = lambda t: get_distribution(t, self.initial_configuration)
        self.distribution_initial = Gaussian(self.initial_configuration["m_0"], self.initial_configuration["P_0"])
        self.distribution_terminal = self.get_distribution(self.total_evolving_time)

    def V_true_fn(self, x: jnp.ndarray): 
        _V_true_fn = lambda x: jnp.dot(x, self.initial_configuration["tilde_F"] @ x) / 2
        _V_true_fn_vmap_x = jax.vmap(_V_true_fn, in_axes=[0])
        if x.ndim == 1:
            return _V_true_fn(x)
        elif x.ndim == 2:
            return _V_true_fn_vmap_x(x)
        else:
            raise ValueError("x should be either 1D (unbatched) or 2D (batched) array.")

    def sample_ground_truth(self, rng, batch_size: int):
        sample_per_time = 100
        assert batch_size >= sample_per_time * 2
        n_random_time = batch_size // sample_per_time

        # sample a single data
        def _sample_ground_truth_fn(rng):
            rng_time, rng_x = jax.random.split(rng, 2)
            # sample time
            t = self.distribution_time.sample(1, rng_time)[0]
            # sample data
            x = self.get_distribution(t).sample(sample_per_time, rng_x)
            return x
        _sample_ground_truth_fn = jax.vmap(_sample_ground_truth_fn, in_axes=[0])
        sample_ground_truth = _sample_ground_truth_fn(jax.random.split(rng, n_random_time))
        sample_shape = sample_ground_truth.shape
        new_first_dim = prod(sample_shape[:2])
        sample_ground_truth = sample_ground_truth.reshape((new_first_dim, *sample_shape[2:]))
        
        return sample_ground_truth
