import jax.numpy as jnp
from core.distribution import Gaussian, Uniform
import jax
from jax.experimental.ode import odeint
from api import ProblemInstance
from utils.common_utils import v_gaussian_score, v_gaussian_log_density
from core.potential import GMMPotential
from utils.sampling_utils import underdamped_langevin_dynamics_scan
from math import prod
import warnings
# To ensure that the coefficient of the Laplacian term in the FPE is 1, L should be sqrt{2}


def initialize_configuration(domain_dim: int, rng):
    gamma_friction = .5
    # number of Gaussians
    n_Gaussian = 3
    # means and covs of Gaussians
    rngs = jax.random.split(rng, n_Gaussian)
    GMM_mean_min = -4
    GMM_mean_max = 4

    P_x_0_scale = 4.

    P_v_0_scale = .1

    m_x_0 = jnp.zeros(domain_dim)
    m_v_0 = jnp.zeros(domain_dim) 

    m_0 = jnp.concatenate([m_x_0, m_v_0])

    P_x_0 = jnp.eye(domain_dim) * P_x_0_scale
    P_v_0 = jnp.eye(domain_dim) * P_v_0_scale

    P_0 = jnp.block([
        [P_x_0, jnp.zeros([domain_dim, domain_dim])],
        [jnp.zeros([domain_dim, domain_dim]), P_v_0]
    ])

    return {
        "gamma_friction": gamma_friction,
        "m_0": m_0,
        "P_0": P_0,

        "GMM": {
            "mus": jnp.stack([jax.random.uniform(_rng, [domain_dim], minval=GMM_mean_min, maxval=GMM_mean_max) for _rng in rngs]),
            # Uniform weights by default
            # "weights": jnp.ones([n_Gaussian]) / n_Gaussian,
        }
    }
    

class KineticFokkerPlanck(ProblemInstance):
    def __init__(self, cfg, rng):
        super().__init__(cfg, rng)
        self.initial_configuration = initialize_configuration(cfg.pde_instance.domain_dim, rng)
        
        # define the potential from configuration
        self.potential = GMMPotential(self.initial_configuration["GMM"]["mus"], jnp.ones([]))
        
        # for the "SDE" sampling scheme, only the initial distribution is available in a closed form
        self.sample_scheme = "SDE"

        self.distribution_initial = Gaussian(self.initial_configuration["m_0"], self.initial_configuration["P_0"])
        
        
    def V_true_fn(self, x: jnp.ndarray): 
        _V_true_fn = self.potential.value
        _V_true_fn_vmap_x = jax.vmap(_V_true_fn, in_axes=[0])
        if x.ndim == 1:
            return _V_true_fn(x)
        elif x.ndim == 2:
            return _V_true_fn_vmap_x(x)
        else:
            raise ValueError("x should be either 1D (unbatched) or 2D (batched) array.")

    def sample_ground_truth(self, rng, batch_size):
        # use Langevin dynamics to sample from the ground truth
        rng, rng2, rng_init, rng_init2, rng_init3 = jax.random.split(rng, 5)
        multiple_init = 30
        multiple_terminal = 30
        n_steps = self.cfg.pde_instance.n_steps
        dt = self.total_evolving_time / n_steps
        # sample q0_p0
        q0_p0 = self.distribution_initial.sample(batch_size, rng_init)
        rngs = jax.random.split(rng, batch_size)
        # maybe have a random initial time offset
        _, sample_0T = underdamped_langevin_dynamics_scan(q0_p0, n_steps, dt, rngs, self.potential.gradient, 
                                                                     self.initial_configuration["gamma_friction"])
        sample_0T = sample_0T.reshape((prod(sample_0T.shape[:2]), *sample_0T.shape[2:]))

        sample_initial = self.distribution_initial.sample(batch_size * multiple_init, rng_init2)
        q0_p02 = self.distribution_initial.sample(batch_size * multiple_terminal, rng_init3)
        rngs2 = jax.random.split(rng2, batch_size * multiple_terminal)
        sample_final, _ = underdamped_langevin_dynamics_scan(q0_p02, n_steps, dt, rngs2, self.potential.gradient, 
                                                                     self.initial_configuration["gamma_friction"])

        return sample_initial, sample_final, sample_0T
    
    # def sample_ground_truth(self, rng, batch_size):
    #     # use Langevin dynamics to sample from the ground truth
    #     rng, rng_init = jax.random.split(rng)
    #     n_steps = self.cfg.pde_instance.n_steps
    #     dt = self.total_evolving_time / n_steps
    #     # sample q0_p0
    #     q0_p0 = self.distribution_initial.sample(batch_size, rng_init)
    #     rngs = jax.random.split(rng, batch_size)
    #     # maybe have a random initial time offset
    #     sample_final, sample_0T = underdamped_langevin_dynamics_scan(q0_p0, n_steps, dt, rngs, self.potential.gradient, 
    #                                                                  self.initial_configuration["gamma_friction"])
    #     sample_0T = sample_0T.reshape((prod(sample_0T.shape[:2]), *sample_0T.shape[2:]))
    #     return q0_p0, sample_final, sample_0T

         
