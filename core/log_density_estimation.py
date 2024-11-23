from core.normalizing_flow import MNF, RealNVP
from api import ProblemInstance
import optax
import jax.random as random
import jax.numpy as jnp
import jax
from functools import partial
import wandb
import matplotlib.pyplot as plt


# Use normalizing flow to model the log-density along the distribution trajectory
def estimate_log_density(cfg, pde_instance: ProblemInstance, rng):
    # batch_size_time = cfg.log_density.train.batch_size_time
    num_epochs = 20000

    rng_keys = ["model_init", "train"]
    rngs = dict(zip(rng_keys, random.split(rng, len(rng_keys))))

    log_density_model = create_normalizing_flow_fn(pde_instance.distribution_initial_x.logdensity, cfg.pde_instance.domain_dim)
    params = log_density_model.init(rngs["model_init"], jnp.array(0.0), jnp.zeros(cfg.pde_instance.domain_dim))

    # create optimizer for RealNVP
    lr = 1e-3    # Initial learning rate
    T0 = 5000     # Steps for initial constant learning rate
    T1 = 15000     # Step at which to end cosine decay and set learning rate to 0

    schedule = create_custom_schedule(lr, T0, T1)
    optimizer = optax.adam(learning_rate=schedule, b1=0.9, eps=1e-4)
    opt_state = optimizer.init(params)

    # density estimation using the offline dataset
    dataset = pde_instance.dataset["0T"] 
    time_grid = pde_instance.dataset["tau_0T"] 

    n_trajectories, n_time_stamps, _ = dataset.shape

    interval_time = 5 # use 1 out of 5 time stamps
    interval_sample = 5 # for a fixed time stamp, use 1 out of 5 samples

    @jax.jit
    def step_fn(opt_state, params, grad):
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    
    @jax.jit
    def value_and_grad_fn(params, data_0T):
        @partial(jax.vmap, in_axes=[None, 0, 0])
        @partial(jax.vmap, in_axes=[None, 0, 0])
        def likelihood_fn(params, t, x):
            return log_density_model.apply(params, t, x)

        def loss_fn(params):
            likelihood = likelihood_fn(params, data_0T["time"], data_0T["data"])
            return -jnp.mean(likelihood)

        v, g = jax.value_and_grad(loss_fn)(params)
        # print(v_and_g)
        return v, g

        

    rng_epoch = random.split(rngs["train"], num_epochs)
    error_per_frequency = 0
    frequency = 100
    for epoch in range(num_epochs):
        # Sample a random batch
        rng_time, rng_sample = random.split(rng_epoch[epoch])
        
        time_index = jnp.arange(n_time_stamps//interval_time) * interval_time
        shift = random.randint(rng_time, [], 0, interval_time)
        random_time_index = time_index + shift

        random_sample_index = random.permutation(rng_sample, jnp.arange(n_trajectories))[:n_trajectories//interval_sample]
        
        data_epoch = dataset[random_sample_index]
        data_epoch = data_epoch[:, random_time_index, :]
        data_epoch = data_epoch[:, :, :cfg.pde_instance.domain_dim]
        tau_epoch = time_grid[random_sample_index]
        tau_epoch = tau_epoch[:, random_time_index]
        data_0T = {"time": tau_epoch, "data": data_epoch}

        # compute gradient
        batch_loss, grad = value_and_grad_fn(params, data_0T)
        # Perform a training step
        params, opt_state = step_fn(opt_state, params, grad)
        error_per_frequency = error_per_frequency + batch_loss
        if (epoch+1) % frequency == 0:
            print(f"Epoch {epoch+1}, Loss: {error_per_frequency/frequency}")
            error_per_frequency = 0
        # print(f"Epoch {epoch+1}, Loss: {batch_loss}")

    @partial(jax.vmap, in_axes=[None, 0])
    def log_density_fn(t, x):
        return log_density_model.apply(params, t, x)
    
    plot_trajectory_of_distributions(log_density_fn)


    return log_density_fn

def create_normalizing_flow_fn(log_prob_0, dim):
    param_dict = {
        'dim': dim,
        'embed_time_dim': 10,
        'couple_mul': 4,
        'mask_type': 'loop',
        'activation_layer': 'celu',
        'soft_init': 1.,
        'ignore_time': False,
    }
    mnf = MNF(**param_dict)
    return RealNVP(mnf, log_prob_0)

def create_custom_schedule(lr, T0, T1):
    # Step 1: Define constant schedule for the first T0 steps
    constant_schedule = optax.constant_schedule(lr)
    
    # Step 2: Define cosine decay schedule from T0 to T1
    cosine_decay_schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr,
        peak_value=lr,          # Start decaying from `lr`
        warmup_steps=0,          # No additional warmup
        decay_steps=(T1 - T0),   # Decay duration
        end_value=lr*1e-2            # Decay to `0.0`
    )

    # Step 3: Join the schedules:
    # - Use `constant_schedule` for first `T0` steps
    # - Use `cosine_decay_schedule` for steps from `T0` to `T1`
    # - Use `constant_schedule` set to `0` for steps beyond `T1`
    schedule = optax.join_schedules(
        schedules=[constant_schedule, cosine_decay_schedule, optax.constant_schedule(lr*1e-2)],
        boundaries=[T0, T1]
    )
    
    return schedule

def plot_trajectory_of_distributions(f, t_min=0, t_max=10, grid_size=100, n_time_points=11):
    """
    Plot the trajectory of 2D distributions over time.

    Args:
        f: Callable, a function that takes (x, y, t) and returns the log density.
        t_min: Minimum time value.
        t_max: Maximum time value.
        grid_size: Resolution of the spatial grid.
        n_time_points: Number of time points to evaluate.
    """
    # Define spatial grid
    x = jnp.linspace(-8, 8, grid_size)
    y = jnp.linspace(-8, 8, grid_size)
    X, Y = jnp.meshgrid(x, y)
    xy = jnp.stack([X.ravel(), Y.ravel()], axis=-1)


    # Define time grid
    time_points = jnp.linspace(t_min, t_max, n_time_points)

    # Create subplots
    fig, axes = plt.subplots(1, n_time_points, figsize=(15, 3), constrained_layout=True)

    for i, t in enumerate(time_points):
        # Compute log density for the grid
        # log_density = jax.vmap(
            # jax.vmap(lambda x: f(t, x), in_axes=(0, None)), in_axes=(None, 0)
        # )(xy)

        log_density = f(t, xy)

        # Convert to density
        density = jnp.exp(log_density).reshape(X.shape)
        # Plot the contours
        ax = axes[i]
        contour = ax.contourf(X, Y, density, levels=50, cmap='viridis')
        ax.set_title(f"t = {t:.1f}")
        ax.axis("off")

    # Add colorbar
    cbar = fig.colorbar(contour, ax=axes, orientation='horizontal', fraction=0.1, pad=0.1)
    cbar.set_label("Density")

    # Save the figure locally
    plt.savefig("trajectory_of_distributions.png")
    plt.show()

    # Log to wandb
    wandb.log({"Density Trajectory": wandb.Image("trajectory_of_distributions.png")})