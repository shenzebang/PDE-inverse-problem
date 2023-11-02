import optax
import wandb
import jax
import jax.numpy as jnp
from utils.common_utils import compute_pytree_norm
from api import Method
import jax.random as random
from optax import GradientTransformation

class JaxTrainer:
    def __init__(self,
                 cfg,
                 method: Method,
                 rng: jnp.ndarray,
                 optimizer: GradientTransformation,
                 forward_fn,
                 params: optax.Params,
                 ):
        self.cfg = cfg
        self.forward_fn = forward_fn
        self.params = params
        self.optimizer = optimizer
        self.method = method
        self.rng = rng

    def fit(self, ):

        # initialize the opt_state
        opt_state = self.optimizer.init(self.params)

        # jit or pmap the gradient computation for efficiency
        def value_and_grad_fn(params, rng):
            return self.method.value_and_grad_fn(self.forward_fn, params, rng)


        if self.cfg.backend.use_pmap_train and jax.local_device_count() > 1:
            _value_and_grad_fn = jax.pmap(value_and_grad_fn, in_axes=(None, 0))

            def value_and_grad_fn_efficient(params, rng):

                rngs = random.split(rng, jax.local_device_count())
                # compute in parallel
                v_g_etc = _value_and_grad_fn(params, rngs)
                v_g_etc = jax.tree_map(lambda _g: jnp.mean(_g, axis=0), v_g_etc)
                return v_g_etc
        else:
            value_and_grad_fn_efficient = jax.jit(value_and_grad_fn)
            # value_and_grad_fn_efficient = value_and_grad_fn

        @jax.jit
        def step(params, opt_state, grad):
            updates, opt_state = self.optimizer.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        @jax.jit
        def test(params, rng):
            return self.method.test_fn(self.forward_fn, params, rng)

        # @jax.jit
        def plot(params, rng):
            return self.method.plot_fn(self.forward_fn, params, rng)

        rngs = jax.random.split(self.rng, self.cfg.train.number_of_iterations)
        for epoch in range(self.cfg.train.number_of_iterations):
            rng = rngs[epoch]
            rng_train, rng_test, rng_plot = random.split(rng, 3)

            v_g_etc = value_and_grad_fn_efficient(self.params, rng_train)
            self.params, opt_state = step(self.params, opt_state, v_g_etc["grad"])

            v_g_etc.pop("grad")
            params_norm = compute_pytree_norm(self.params)
            v_g_etc["params_norm"] = params_norm
            wandb.log(v_g_etc, step=epoch)
            if (epoch % self.cfg.test.frequency == 0 and self.method.test_fn is not None) or epoch >= self.cfg.train.number_of_iterations - 3:
                result_epoch = test(self.params, rng_test)
                wandb.log(result_epoch, step=epoch)
                if self.cfg.test.verbose:
                    msg = f"In epoch {epoch + 1: 5d}, "
                    for key in v_g_etc:
                        msg = msg + f"{key} is {v_g_etc[key]: .3e}, "
                    for key in result_epoch:
                        msg = msg + f"{key} is {result_epoch[key]: .3e}, "
                    print(msg)

            if (epoch + 1) % self.cfg.plot.frequency == 0 and self.method.plot_fn is not None: plot(self.params, rng_plot)

        return self.params
