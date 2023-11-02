import jax.numpy as jnp
import jax
from flax import linen as nn
from typing import Tuple

# for DEBUG
from example_problems.fokker_planck_example import initialize_configuration

class V_hypothesis(nn.Module):
    output_dim: int 
    hidden_dims: Tuple[int]
    def setup(self):
        # self.layers = [nn.Dense(dim_out, kernel_init=nn.initializers.xavier_uniform()) for dim_out in list(self.hidden_dims) + [self.output_dim]]
        self.layers = [nn.Dense(dim_out, kernel_init=nn.initializers.kaiming_normal()) for dim_out in
                       list(self.hidden_dims) + [self.output_dim]]

        # self.act = jax.nn.sigmoid
        self.act = jax.nn.tanh
        # self.act = jax.nn.relu

    def __call__(self, x: jnp.ndarray):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)

        return x
    
class V_hypothesis_DEBUG(nn.Module):
    output_dim: int 
    hidden_dims: Tuple[int]
    def setup(self):
        # self.layers = [nn.Dense(dim_out, kernel_init=nn.initializers.xavier_uniform()) for dim_out in list(self.hidden_dims) + [self.output_dim]]
        self.layers = [nn.Dense(dim_out, kernel_init=nn.initializers.kaiming_normal()) for dim_out in
                       list(self.hidden_dims) + [self.output_dim]]

        # self.act = jax.nn.sigmoid
        self.act = jax.nn.tanh
        self.initial_configuration = initialize_configuration(4)

    def __call__(self, y: jnp.ndarray):

        for i, layer in enumerate(self.layers):
            x = layer(y)
            if i < len(self.layers) - 1:
                x = self.act(x)


        return self.V_true_fn(y)[None]

    def V_true_fn(self, x: jnp.ndarray): 
        _V_true_fn = lambda x: jnp.dot(x, self.initial_configuration["F"] @ x) / 2
        _V_true_fn_vmap_x = jax.vmap(_V_true_fn, in_axes=[0])
        if x.ndim == 1:
            return _V_true_fn(x)
        elif x.ndim == 2:
            return _V_true_fn_vmap_x(x)
        else:
            raise ValueError("x should be either 1D (unbatched) or 2D (batched) array.")


def get_model(cfg, DEBUG=False, pde_instance=None):
    if cfg.neural_network.n_resblocks > 0:
        # use resnet
        raise NotImplementedError
    else:
        # do not use resnet
        if not DEBUG:
            model = V_hypothesis(output_dim=1,
                          hidden_dims=[cfg.neural_network.hidden_dim] * cfg.neural_network.layers
                          )
        else:
            model = V_hypothesis_DEBUG(output_dim=1, hidden_dims=[cfg.neural_network.hidden_dim] * cfg.neural_network.layers)

        return model

