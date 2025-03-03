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
from flax import linen as nn

# To ensure that the coefficient of the Laplacian term in the FPE is 1, L should be sqrt{2}


def initialize_configuration(domain_dim: int, rng):
    gamma_friction = 0.5
    # number of Gaussians
    n_Gaussian = 3
    # means and covs of Gaussians
    rngs = jax.random.split(rng, n_Gaussian)
    GMM_mean_min = -4
    GMM_mean_max = 4

    P_x_0_scale = 4.0

    P_v_0_scale = 0.1

    m_x_0 = jnp.zeros(domain_dim)
    m_v_0 = jnp.zeros(domain_dim)

    m_0 = jnp.concatenate([m_x_0, m_v_0])

    P_x_0 = jnp.eye(domain_dim) * P_x_0_scale
    P_v_0 = jnp.eye(domain_dim) * P_v_0_scale

    P_0 = jnp.block(
        [
            [P_x_0, jnp.zeros([domain_dim, domain_dim])],
            [jnp.zeros([domain_dim, domain_dim]), P_v_0],
        ]
    )

    return {
        "n_Gaussian": n_Gaussian,
        "gamma_friction": gamma_friction,
        "m_0": m_0,
        "P_0": P_0,
        "m_x_0": m_x_0,
        "P_x_0": P_x_0,
        "GMM": {
            "mus": jnp.stack(
                [
                    jax.random.uniform(
                        _rng, [domain_dim], minval=GMM_mean_min, maxval=GMM_mean_max
                    )
                    for _rng in rngs
                ]
            ),
            # Uniform weights by default
            # "weights": jnp.ones([n_Gaussian]) / n_Gaussian,
        },
    }


class KineticFokkerPlanck(ProblemInstance):
    def __init__(self, cfg, rng):
        super().__init__(cfg, rng)
        rng_initial_config, rng_dataset = jax.random.split(rng)

        self.initial_configuration = initialize_configuration(
            cfg.pde_instance.domain_dim, rng_initial_config
        )

        # define the potential from configuration
        self.potential = GMMPotential(
            self.initial_configuration["GMM"]["mus"], jnp.ones([])
        )

        # for the "SDE" sampling scheme, only the initial distribution is available in a closed form
        self.sample_scheme = "SDE"
        self.sample_mode = self.cfg.pde_instance.sample_mode

        self.distribution_initial = Gaussian(
            self.initial_configuration["m_0"], self.initial_configuration["P_0"]
        )
        self.distribution_initial_x = Gaussian(
            self.initial_configuration["m_x_0"], self.initial_configuration["P_x_0"]
        )

        if self.sample_mode == "offline":
            self.dataset = self.generate_ground_truth_dataset(rng_dataset)

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
        _, sample_0T = underdamped_langevin_dynamics_scan(
            q0_p0,
            n_steps,
            dt,
            rngs,
            self.potential.gradient,
            self.initial_configuration["gamma_friction"],
        )
        # flatten the tensor from [n_steps, batch_size, 2*dim] to [n_steps*batch_size, 2*dim]
        sample_0T = sample_0T.reshape((prod(sample_0T.shape[:2]), *sample_0T.shape[2:]))

        sample_initial = self.distribution_initial.sample(
            batch_size * multiple_init, rng_init2
        )
        q0_p02 = self.distribution_initial.sample(
            batch_size * multiple_terminal, rng_init3
        )
        rngs2 = jax.random.split(rng2, batch_size * multiple_terminal)
        sample_final, _ = underdamped_langevin_dynamics_scan(
            q0_p02,
            n_steps,
            dt,
            rngs2,
            self.potential.gradient,
            self.initial_configuration["gamma_friction"],
        )

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

    def generate_ground_truth_dataset(self, rng):
        rng_initial, rng_terminal, rng_0T = jax.random.split(rng, 3)

        # generate sample_initial; Shape [sample_initial_size, 2*dim]
        dataset = {
            "initial": self.distribution_initial.sample(
                self.cfg.pde_instance.sample_initial_size, rng_initial
            ),
        }

        # configs for sample_terminal; Shape [sample_terminal_size, 2*dim]
        rng_terminal_0, rng_terminal_1 = jax.random.split(rng_terminal)
        n_steps = self.cfg.pde_instance.n_steps_terminal
        dt = self.total_evolving_time / n_steps
        q0_p0 = self.distribution_initial.sample(
            self.cfg.pde_instance.sample_terminal_size, rng_terminal_0
        )
        rngs = jax.random.split(
            rng_terminal_1, self.cfg.pde_instance.sample_terminal_size
        )
        dataset["terminal"], _, _ = underdamped_langevin_dynamics_scan(
            q0_p0,
            n_steps,
            dt,
            rngs,
            self.potential.gradient,
            self.initial_configuration["gamma_friction"],
        )

        # configs for sample_OT; Shape [n_steps, sample_0T_size, 2*dim]
        rng_0T_0, rng_0T_1 = jax.random.split(rng_0T)
        n_steps = self.cfg.pde_instance.n_steps_0T
        dt = self.total_evolving_time / n_steps
        q0_p0 = self.distribution_initial.sample(
            self.cfg.pde_instance.sample_0T_size, rng_0T_0
        )
        rngs = jax.random.split(rng_0T_1, self.cfg.pde_instance.sample_0T_size)
        _, dataset["0T"], dataset["tau_0T"] = underdamped_langevin_dynamics_scan(
            q0_p0,
            n_steps,
            dt,
            rngs,
            self.potential.gradient,
            self.initial_configuration["gamma_friction"],
        )

        return dataset

    def create_parametric_model(self):
        model = V_parametric(
            dim=self.dim, n_Gaussians=self.initial_configuration["n_Gaussian"]
        )

        return model


class V_parametric(nn.Module):
    dim: int
    n_Gaussians: int

    def setup(self):
        # The GMM has uniform weights for each Gaussian, and each Gaussian has only the mean as variable.
        # Create a one-layer MLP for every Gaussian
        # self.mus = [nn.Dense(self.dim) for _ in range(self.n_Gaussians)]
        self.mus = self.param(
            "mus",  # Name of the variable
            lambda rng, shape: jax.random.normal(rng, shape),  # Initialization function
            (
                self.n_Gaussians,
                self.dim,
            ),  # Shape of the variable
        )

    def __call__(self, y_input: jnp.ndarray):
        potential = GMMPotential(self.mus, jnp.ones([]))
        # evaluate the GMM potential
        return potential.value(y_input)[None]
