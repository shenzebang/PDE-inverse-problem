import jax.numpy as jnp
from core.distribution import Gaussian, Uniform
import jax
from jax.experimental.ode import odeint
from api import ProblemInstance
from utils.common_utils import v_gaussian_score, v_gaussian_log_density
from core.potential import QuadraticPotential

# OU process
# dX(t) = -FX(t) dt + \sqrt{L} dW(t)
# Assume F is positive definite
# If X(0) \sim N(m(0), P(0)), we have X(t) \sim N(m(t), P(t))
# where m(t) = exp(-Ft) m(0), P(t) = exp(-Ft)P(0)exp(-Ft) + \int_0^t exp(-F(t-s)) L exp(-F(t-s)) ds

# For the simplicity of computation, let F = USU' be the SVD of F.
# Denote B = U'LU

# To ensure that the coefficient of the Laplacian term in the FPE is 1, L should be sqrt{2}

def initialize_configuration(domain_dim: int):
    F_scale = 1.
    L_scale = 2
    m_0_scale = 1.
    P_0_scale = 5.

    m_0 = jnp.ones(domain_dim) * m_0_scale
    P_0 = jnp.eye(domain_dim) * P_0_scale
    # F = jnp.eye(domain_dim) * F_scale
    _F = jax.random.normal(jax.random.PRNGKey(2217), (domain_dim, domain_dim + 1))
    F = _F @ _F.transpose() * F_scale
    # _L = jax.random.normal(jax.random.PRNGKey(2219), (domain_dim, domain_dim + 1))
    # L = _L @ _L.transpose() * L_scale
    L = jnp.eye(domain_dim) * L_scale
    U, s, _ = jnp.linalg.svd(F)
    
    return {
        "F": F,
        "L": L,
        "U": U,
        "ss": s + s[:, None],
        "B": U.transpose() @ L @ U,
        "B_0": U.transpose() @ P_0 @ U,
        "s": s,
        "m_0": m_0,
        "P_0": P_0,
    }

def OU_process(t, configuration):
    exp_t_s = jnp.diag(jnp.exp(- t * configuration["s"]))
    m_t = configuration["U"] @ exp_t_s @ configuration["U"].transpose() @ configuration["m_0"]
    P_t_1 =  exp_t_s @ configuration["B_0"] @ exp_t_s 
    B_S = configuration["B"] / configuration["ss"]
    P_t_2 = B_S - exp_t_s @ B_S @ exp_t_s
    P_t = configuration["U"] @ (P_t_1 + P_t_2) @ configuration["U"].transpose()
    return m_t, P_t

OU_process_vmapt = jax.vmap(OU_process, in_axes=[0, None])

def get_distribution(t: jnp.ndarray, configuration):
    mean, cov = OU_process(t, configuration)
    return Gaussian(mean, cov)

class FokkerPlanck(ProblemInstance):
    def __init__(self, cfg, rng):
        super().__init__(cfg, rng)
        self.initial_configuration = initialize_configuration(cfg.pde_instance.domain_dim)
        self.get_distribution = lambda t: get_distribution(t, self.initial_configuration)
        self.distribution_initial = self.get_distribution(jnp.zeros([]))
        self.distribution_terminal = self.get_distribution(self.total_evolving_time)

        # For DEBUG #
        # test_OU(self.total_evolving_time, self.initial_configuration)

    def V_true_fn(self, x: jnp.ndarray): 
        _V_true_fn = lambda x: jnp.dot(x, self.initial_configuration["F"] @ x) / 2
        _V_true_fn_vmap_x = jax.vmap(_V_true_fn, in_axes=[0])
        if x.ndim == 1:
            return _V_true_fn(x)
        elif x.ndim == 2:
            return _V_true_fn_vmap_x(x)
        else:
            raise ValueError("x should be either 1D (unbatched) or 2D (batched) array.")

    def sample_ground_truth(self, rng, batch_size: int):
        # sample a single data
        def _sample_ground_truth(rng):
            rng_time, rng_x = jax.random.split(rng, 2)
            # sample time
            t = self.distribution_time.sample(1, rng_time)[0]
            # sample data
            x = self.get_distribution(t).sample(1, rng_x)[0]
            return x
        _sample_ground_truth = jax.vmap(_sample_ground_truth, in_axes=[0])
        
        # sample_ground_truth_batch = jax.vmap(sample_ground_truth_single, in_axes=[0])
        return _sample_ground_truth(jax.random.split(rng, batch_size))

# A validating module to test the correctness of the above closed form
# The dynamics of m_t and P_t are
# dm/dt = -F m, dP/dt = -FP - PF + L 
def test_OU(total_evolving_time: jnp.ndarray, configuration):
    t_space = jnp.linspace(0., total_evolving_time, 101)
    # compute the mean and variance according to the ODE
    state_0 = {"m": configuration["m_0"], "P": configuration["P_0"]}
    def ode_func(states, t):
        return {
            "m": -configuration["F"] @ states["m"],
            "P": -configuration["F"] @ states["P"] - states["P"] @ configuration["F"] + configuration["L"],
        }
    
    state_T = odeint(ode_func, state_0, t_space)
    # compute the mean and variance according to the closed form solution
    means, covs = OU_process_vmapt(t_space, configuration)
    mean_error = jnp.mean(jnp.sum((state_T["m"] - means) ** 2, axis=-1))
    cov_error = jnp.mean(jnp.sum((state_T["P"] - covs) ** 2, axis=(-2,-1)))
    print(f"error of means {mean_error}, error of covs {cov_error}.")
