import jax.numpy as jnp
import jax
from flax import linen as nn
from typing import Tuple
from api import ProblemInstance

# for DEBUG
from example_problems.fokker_planck_example import initialize_configuration

class V_hypothesis_backup(nn.Module):
    output_dim: int 
    hidden_dims: Tuple[int]
    def setup(self):
        # self.layers = [nn.Dense(dim_out, kernel_init=nn.initializers.xavier_uniform()) for dim_out in list(self.hidden_dims) + [self.output_dim]]
        self.layers = [nn.Dense(dim_out, kernel_init=nn.initializers.kaiming_normal()) for dim_out in
                       list(self.hidden_dims) + [self.output_dim]]

        # self.act = jax.nn.sigmoid
        self.act = jax.nn.tanh
        # self.act = jax.nn.relu

    def __call__(self, x_0: jnp.ndarray):
        # x_0 = jnp.concatenate([x_0, jnp.outer(x_0, x_0).flatten()], axis=-1)
        x = x_0
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)

        return x

class V_hypothesis(nn.Module):
    output_dim: int 
    hidden_dims: Tuple[int]
    def setup(self):
        # self.layers = [nn.Dense(dim_out, kernel_init=nn.initializers.xavier_uniform()) for dim_out in list(self.hidden_dims) + [self.output_dim]]
        # self.layers = [nn.Dense(dim_out, kernel_init=nn.initializers.kaiming_normal()) for dim_out in [self.output_dim]]
        self.F = nn.Dense(4)
        # self.layers = [nn.Dense(dim_out, kernel_init=nn.initializers.kaiming_normal()) for dim_out in
                    #    list(self.hidden_dims) + [self.output_dim]]

        self.layers = [nn.Dense(dim_out, kernel_init=nn.initializers.kaiming_normal()) for dim_out in
                       list(self.hidden_dims) + [40]]

        # self.act = jax.nn.sigmoid
        self.act = jax.nn.tanh
        # self.act = jax.nn.sigmoid
        # self.act = jax.nn.tanh
        self.initial_configuration = initialize_configuration(4)

    def __call__(self, y_input: jnp.ndarray):
        # return jnp.sum(y * self.F(y), axis=-1)[None]
        # y = jnp.concatenate([y_input, jnp.outer(y_input, y_input).flatten()], axis=-1)
        x = y_input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)


        # return x + jnp.sum(y_input * self.F(y_input), axis=-1)[None]
        return jnp.sum(x**2, axis=-1)[None]

class V_hypothesis_DEBUG(nn.Module):
    output_dim: int 
    hidden_dims: Tuple[int]
    pde_instance: ProblemInstance

    def setup(self):
        # self.layers = [nn.Dense(dim_out, kernel_init=nn.initializers.xavier_uniform()) for dim_out in list(self.hidden_dims) + [self.output_dim]]
        # self.layers = [nn.Dense(dim_out, kernel_init=nn.initializers.kaiming_normal()) for dim_out in [self.output_dim]]
        self.F = nn.Dense(self.pde_instance.dim)
        # self.layers = [nn.Dense(dim_out, kernel_init=nn.initializers.kaiming_normal()) for dim_out in
                    #    list(self.hidden_dims) + [self.output_dim]]

        self.layers = [nn.Dense(dim_out, kernel_init=nn.initializers.kaiming_normal()) for dim_out in
                       list(self.hidden_dims) + [40]]

        # self.act = jax.nn.sigmoid
        self.act = jax.nn.tanh
        # self.act = jax.nn.sigmoid
        # self.act = jax.nn.tanh
        self.initial_configuration = initialize_configuration(4)

    def __call__(self, y_input: jnp.ndarray):
        return jnp.sum(y_input * self.F(y_input), axis=-1)[None]
        # # y = jnp.concatenate([y_input, jnp.outer(y_input, y_input).flatten()], axis=-1)
        # x = y_input
        # for i, layer in enumerate(self.layers):
        #     x = layer(x)
        #     if i < len(self.layers) - 1:
        #         x = self.act(x)


        # # return x + jnp.sum(y_input * self.F(y_input), axis=-1)[None]
        # return jnp.sum(x**2, axis=-1)[None]

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
    if cfg.estimation_mode == "parametric":
        print("----Using parametric model----")
        model = pde_instance.create_parametric_model()
        return model
    elif cfg.estimation_mode == "non-parametric":
        print("----Using non-parametric model----")
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
                model = V_hypothesis_DEBUG(output_dim=1, hidden_dims=[cfg.neural_network.hidden_dim] * cfg.neural_network.layers,
                                        pde_instance=pde_instance)

            return model
    else:
        raise NotImplementedError

