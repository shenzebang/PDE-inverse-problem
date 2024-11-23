import jax
import jax.numpy as jnp
from jax import random, jit
from jax.lax import scan
from functools import partial
def update_step(state, dt, potential_grad, gamma_friction):
    q, p, key = state
    key, subkey = random.split(key)

    # Compute the gradient of the potential
    grad_U = potential_grad(q)
    
    # Gaussian noise term
    noise = jnp.sqrt(2) * random.normal(subkey, p.shape)
    
    # Update the momentum
    p_new = p - dt * grad_U + jnp.sqrt(dt) * noise - gamma_friction * p * dt
    
    # Update the position
    q_new = q + dt * p_new
    
    return (q_new, p_new, key), jnp.concatenate([q_new, p_new])


@partial(jax.vmap, in_axes=[0, None, None, 0, None, None])
def underdamped_langevin_dynamics_scan(q0_p0, n_steps, dt, key, potential_grad, gamma_friction):
    key, key_init = random.split(key, 2)
    q0, p0 = jnp.split(q0_p0, 2)
    initial_state = (q0, p0, key)

    # a random time shift to ensure that all the time stamps can be accessed
    tau_0 = random.uniform(key_init, []) * dt # initial_time_shift
    state_after_shift, sample_after_shift = update_step(initial_state, tau_0, potential_grad, gamma_friction)

    
    # Run the scan and collect the trajectory
    final_state, trajectory = scan(
        lambda state, _: update_step(state, dt, potential_grad, gamma_friction),
        state_after_shift,
        None,
        length=n_steps-1
    )
    
    # get the final sample (note that the above computation only covers tau_0 + (n_steps-1) *dt, while we need n_steps * dt)
    tau_end = dt - tau_0
    _, last_sample = update_step(final_state, tau_end, potential_grad, gamma_friction)

    tau_trajectory = tau_0 + jnp.arange(n_steps) * dt # the time stamp at which the sample is taken

    # trajectory is a tuple of (positions, momenta)
    # append the initial sample to trajectory
    return last_sample, jnp.concatenate([jnp.expand_dims(sample_after_shift, 0), trajectory]), tau_trajectory
    

