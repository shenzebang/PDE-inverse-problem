import jax
import jax.numpy as jnp
from example_problems.kinetic_mckean_vlasov_example_quadratic import KineticMcKeanVlasov
from core.model import get_model
from utils.common_utils import compute_pytree_norm, hessian_vector_product
import jax.random as random
import optax
from functools import partial


def value_and_grad_fn(forward_fn, params, data, rng, pde_instance: KineticMcKeanVlasov):
    # split the data into x ([n, n_time, d]) and v ([n, n_time, d]),
    # TODO: x ([n*n_time, d])!!!
    x_0T, v_0T = jnp.split(data["0T"], 2, axis=-1)
    # ref_0T, _ = jnp.split(data["ref"], 2, axis=-1)
    # ref_0T, _ = jnp.split(data["0T"], 2, axis=-1)  # TODO: using 0T data for now.
    tau_0T = data["tau_0T"]
    x_0T = jnp.reshape(x_0T, (-1, tau_0T.shape[0], x_0T.shape[-1]))
    v_0T = jnp.reshape(v_0T, (-1, tau_0T.shape[0], v_0T.shape[-1]))
    ref_0T = x_0T

    # x is of size [n, n_time, d], ref is of size [m, n_time, d]. x-ref is of size [m, n, n_time, d]
    x_minus_ref = jnp.expand_dims(x_0T, axis=0) - jnp.expand_dims(ref_0T, axis=1)

    # Define the functions
    Phi = lambda x, params: forward_fn(params, x)[0]
    # Phi = lambda x, params: pde_instance.Phi_true_fn(x)

    nabla_Phi = jax.grad(Phi, argnums=0)

    def my_prod(x, v, params):
        f = lambda x: Phi(x, params)
        return jnp.dot(v, hessian_vector_product(f, x, v))

    # When evaluating value, the input shape is x: [m, n, n_time, d]
    Phi_v = jax.vmap(Phi, in_axes=[0, None])  # vmap for n_time
    Phi_v = jax.vmap(Phi_v, in_axes=[0, None])  # vmap for n
    Phi_v = jax.vmap(Phi_v, in_axes=[0, None])  # vmap for m

    # When evaluating gradient, the input shape is x: [m, n, n_time, d]
    nabla_Phi_v = jax.vmap(nabla_Phi, in_axes=[0, None])  # vmap for n_time
    nabla_Phi_v = jax.vmap(nabla_Phi_v, in_axes=[0, None])  # vmap for n
    nabla_Phi_v = jax.vmap(nabla_Phi_v, in_axes=[0, None])  # vmap for m

    # When evaluating Hessian, the input shape is x: [m, n, n_time, d] & v: [n, n_time, d]
    my_prod = jax.vmap(my_prod, in_axes=[0, 0, None])  # vmap for n_time
    my_prod = jax.vmap(my_prod, in_axes=[0, 0, None])  # vmap for n
    my_prod = jax.vmap(my_prod, in_axes=[0, None, None])  # vmap for m

    Phi_true = lambda x: pde_instance.Phi_true_fn(x)
    nabla_Phi_true = jax.grad(Phi_true, argnums=0)
    # When evaluating the true gradient, the input shape is x: [m, n, n_time, d]
    nabla_Phi_true = jax.vmap(nabla_Phi_true, in_axes=[0])  # vmap for n_time
    nabla_Phi_true = jax.vmap(nabla_Phi_true, in_axes=[0])  # vmap for n
    nabla_Phi_true = jax.vmap(nabla_Phi_true, in_axes=[0])  # vmap for m

    partial_s_log_density = jax.vmap(
        pde_instance.partial_s_log_density_fn, in_axes=[0, 1]
    )(
        tau_0T, x_0T
    )  # shape: [n_time, n]
    partial_s_log_density = partial_s_log_density.reshape(
        -1, tau_0T.shape[0]
    )  # shape: [n, n_time]
    partial_s2_log_density = jax.vmap(
        pde_instance.partial_s2_log_density_fn, in_axes=[0, 1]
    )(
        tau_0T, x_0T
    )  # shape: [n_time, n]
    partial_s2_log_density = partial_s2_log_density.reshape(
        -1, tau_0T.shape[0]
    )  # shape: [n, n_time]

    def loss_fn(params):
        loss_nabla = nabla_Phi_v(x_minus_ref, params)
        loss_nabla = jnp.mean(loss_nabla, axis=0)  # average of m
        loss_nabla = jnp.sum(loss_nabla**2, axis=-1)  # compute the square norm
        loss_nabla = jnp.mean(loss_nabla)  # average of n and n_time

        loss_Hessian = my_prod(x_minus_ref, v_0T, params)
        loss_Hessian = jnp.mean(loss_Hessian, axis=0)  # average of m
        loss_Hessian = jnp.mean(loss_Hessian)  # average of n and n_time

        loss_value = Phi_v(x_minus_ref, params)
        loss_value = jnp.mean(loss_value, axis=0)  # average of m
        loss_value = loss_value * (
            partial_s2_log_density
            + partial_s_log_density**2
            + pde_instance.initial_configuration["gamma_friction"]
            * partial_s_log_density
        )
        loss_value = jnp.mean(loss_value)  # average of n and n_time

        loss_nabla_true = jnp.mean(
            jnp.sum(jnp.mean(nabla_Phi_true(x_minus_ref), axis=0) ** 2, axis=-1)
        )
        return loss_nabla - 2 * loss_Hessian + 2 * loss_value + loss_nabla_true

    def loss_ground_truth_fn(params):
        return jnp.mean(
            jnp.sum(
                (
                    jnp.mean(nabla_Phi_true(x_minus_ref), axis=0)
                    - jnp.mean(nabla_Phi_v(x_minus_ref, params), axis=0)
                )
                ** 2,
                axis=-1,
            )
        )

    vg_fn = jax.value_and_grad(loss_fn)
    loss, grad = vg_fn(params)
    grad_norm = compute_pytree_norm(grad)

    return {
        "loss": loss,
        "grad": grad,
        "grad_norm": grad_norm,
        "loss ground truth": loss_ground_truth_fn(params),
    }


def test_fn(forward_fn, pde_instance: KineticMcKeanVlasov, rng):
    # V = lambda x: forward_fn(x)[0]
    # nabla_V = jax.grad(V, argnums=0)
    # nabla_V_vmap_x = jax.vmap(nabla_V, in_axes=[0])

    # V_true = lambda x: pde_instance.V_true_fn(x)
    # nabla_V_true = jax.grad(V_true, argnums=0)
    # nabla_V_true_vmap_x = jax.vmap(nabla_V_true, in_axes=[0])

    # # sample data from initial and terminal distributions
    # rng_initial, rng_terminal = jax.random.split(rng, 2)
    # data_initial = pde_instance.distribution_initial.sample(10000, rng_initial)
    # data_terminal = pde_instance.distribution_terminal.sample(10000, rng_terminal)
    # # test function value and gradient
    # nabla_V_pred_initial, nabla_V_pred_terminal = nabla_V_vmap_x(data_initial), nabla_V_vmap_x(data_terminal)
    # nabla_V_true_initial, nabla_V_true_terminal = nabla_V_true_vmap_x(data_initial), nabla_V_true_vmap_x(data_terminal)
    # relative_l2_error_initial = jnp.sqrt(jnp.mean(jnp.sum((nabla_V_pred_initial - nabla_V_true_initial) ** 2, axis=-1))/jnp.mean(jnp.sum(nabla_V_true_initial ** 2, axis=-1)))
    # relative_l2_error_terminal = jnp.sqrt(jnp.mean(jnp.sum((nabla_V_pred_terminal - nabla_V_true_terminal) ** 2, axis=-1))/jnp.mean(jnp.sum(nabla_V_true_terminal ** 2, axis=-1)))
    # return {"relative error of gradient estimation initial": relative_l2_error_initial,
    #         "relative error of gradient estimation terminal": relative_l2_error_terminal,}
    return {}
    # return {"KL": KL, "Fisher Information": Fisher_information}


def create_model_fn(pde_instance: KineticMcKeanVlasov):
    net = get_model(pde_instance.cfg, DEBUG=False, pde_instance=pde_instance)
    x, v = jnp.split(
        pde_instance.distribution_initial.sample(1, random.PRNGKey(1))[0],
        indices_or_sections=2,
        axis=-1,
    )
    params = net.init(random.PRNGKey(11), x)
    return net, params
