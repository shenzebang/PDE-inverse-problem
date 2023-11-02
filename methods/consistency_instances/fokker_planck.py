import jax
import jax.numpy as jnp
from example_problems.fokker_planck_example import FokkerPlanck
from core.model import get_model
from utils.common_utils import compute_pytree_norm
import jax.random as random
import optax

# def value_and_grad_fn(forward_fn, params, data, rng, config, pde_instance: FokkerPlanck):
#     V = lambda x, params: forward_fn(params, x)[0]
#     V_true = lambda x: pde_instance.V_true_fn(x)
#     nabla_V = jax.grad(V, argnums=0)
#     nabla_V_true = jax.grad(V_true, argnums=0)
    
    
#     V_vmap_x = jax.vmap(V, in_axes=[0, None])
#     nabla_V_vmap_x = jax.vmap(nabla_V, in_axes=[0, None])
#     nabla_V_true_vmap_x = jax.vmap(nabla_V_true, in_axes=[0])
    

#     def loss_fn(params):
#         nabla_V_true_vmap_x(data["0T"])
#         nabla_V_vmap_x(data["0T"], params)
#         loss_nabla = jnp.mean(jnp.sum((nabla_V_true_vmap_x(data["0T"]) - nabla_V_vmap_x(data["0T"], params))**2, axis=-1))
#         # loss_value = jnp.mean((V_vmap_x(data["0T"], params) - V_true(data["0T"]))**2)
#         return loss_nabla 
    
#     vg_fn = jax.value_and_grad(loss_fn)
#     loss, grad = vg_fn(params)    
#     grad_norm = compute_pytree_norm(grad)
#     return {"loss": loss, "grad": grad, "grad_norm": grad_norm}

def value_and_grad_fn(forward_fn, params, data, rng, pde_instance: FokkerPlanck):
    V = lambda x, params: forward_fn(params, x)[0]
    nabla_V = jax.grad(V, argnums=0)
    hessian_V = jax.jacfwd(nabla_V, argnums=0)
    laplacian_V = lambda x, params: jnp.sum(jnp.diag(hessian_V(x, params)))
    
    V_map_x = jax.vmap(V, in_axes=[0, None])
    nabla_V_vmap_x = jax.vmap(nabla_V, in_axes=[0, None])
    laplacian_V_vmap_x =jax.vmap(laplacian_V, in_axes=[0, None])
    
    V_true = lambda x: pde_instance.V_true_fn(x)
    nabla_V_true = jax.grad(V_true, argnums=0)
    nabla_V_true_vmap_x = jax.vmap(nabla_V_true, in_axes=[0])

    def loss_fn(params):
        loss_initial = jnp.mean(V_map_x(data["initial"], params))
        loss_terminal = jnp.mean(V_map_x(data["terminal"], params))
        loss_nabla = jnp.mean(jnp.sum(nabla_V_vmap_x(data["0T"], params) ** 2, axis=-1))
        loss_laplacian = jnp.mean(laplacian_V_vmap_x(data["0T"], params))
        loss_nabla_true = jnp.mean(jnp.sum(nabla_V_true_vmap_x(data["0T"])**2, axis=-1))
        return (loss_nabla - 2 * loss_laplacian + loss_nabla_true) + (2 * loss_terminal - 2 * loss_initial)/pde_instance.total_evolving_time
    
    vg_fn = jax.value_and_grad(loss_fn)
    loss, grad = vg_fn(params)    
    grad_norm = compute_pytree_norm(grad)
    return {"loss": loss, "grad": grad, "grad_norm": grad_norm}


def test_fn(forward_fn, pde_instance: FokkerPlanck, rng):
    V = lambda x: forward_fn(x)[0]
    nabla_V = jax.grad(V, argnums=0)
    nabla_V_vmap_x = jax.vmap(nabla_V, in_axes=[0])
    
    V_true = lambda x: pde_instance.V_true_fn(x)
    nabla_V_true = jax.grad(V_true, argnums=0)
    nabla_V_true_vmap_x = jax.vmap(nabla_V_true, in_axes=[0])

    # sample data from initial and terminal distributions
    rng_initial, rng_terminal = jax.random.split(rng, 2)
    data_initial = pde_instance.distribution_initial.sample(10000, rng_initial)
    data_terminal = pde_instance.distribution_terminal.sample(10000, rng_terminal)
    # test function value and gradient
    nabla_V_pred_initial, nabla_V_pred_terminal = nabla_V_vmap_x(data_initial), nabla_V_vmap_x(data_terminal)
    nabla_V_true_initial, nabla_V_true_terminal = nabla_V_true_vmap_x(data_initial), nabla_V_true_vmap_x(data_terminal)
    relative_l2_error_initial = jnp.sqrt(jnp.mean(jnp.sum((nabla_V_pred_initial - nabla_V_true_initial) ** 2, axis=-1))/jnp.mean(jnp.sum(nabla_V_true_initial ** 2, axis=-1)))
    relative_l2_error_terminal = jnp.sqrt(jnp.mean(jnp.sum((nabla_V_pred_terminal - nabla_V_true_terminal) ** 2, axis=-1))/jnp.mean(jnp.sum(nabla_V_true_terminal ** 2, axis=-1)))
    return {"relative error of gradient estimation initial": relative_l2_error_initial,
            "relative error of gradient estimation terminal": relative_l2_error_terminal,}

    # return {"KL": KL, "Fisher Information": Fisher_information}

def create_model_fn(pde_instance: FokkerPlanck):
    # net = KiNet(time_embedding_dim=20, append_time=False)
    net = get_model(pde_instance.cfg)
    params = net.init(random.PRNGKey(11), pde_instance.distribution_initial.sample(1, random.PRNGKey(1)))
    # params = net.init(random.PRNGKey(11), jnp.zeros(1),
    #                   jnp.squeeze(pde_instance.distribution_0.sample(1, random.PRNGKey(1))))

    print("Pretraining the potential function to improve the performance.")
    params = potential_pretraining(pde_instance=pde_instance, net=net, params=params)
    print("Finished pretraining.")

    return net, params


def potential_pretraining(pde_instance: FokkerPlanck, net, params):
    # create an optimizer for pretrain
    optimizer = optax.chain(optax.clip(1),
                                    optax.add_decayed_weights(1e-4),
                                    optax.sgd(learning_rate=1e-3, momentum=0.9)
                                    )
    opt_state = optimizer.init(params)

    pretrain_steps = 4096
    # pretrain using the initial data
    key_pretrains = random.split(random.PRNGKey(2199), pretrain_steps)
    
    V_true = lambda x: pde_instance.V_true_fn(x)
    nabla_V_true = jax.grad(V_true, argnums=0)
    perturbation_rate = .1
    V = lambda x, params: net.apply(params, x)[0]
    nabla_V = jax.grad(V, argnums=0)

    def pretrain_loss_fn(params, data):
        return jnp.mean(jnp.sum((nabla_V(data, params) - perturbation_rate * nabla_V_true(data)) ** 2, axis=-1))
    
    pretrain_loss_fn = jax.vmap(pretrain_loss_fn, in_axes=[None, 0])

    def loss_fn(params, data):
        return jnp.mean(pretrain_loss_fn(params, data))
    
    grad_fn = jax.grad(loss_fn, argnums=0)

    def update(key_pretrain, opt_state, params):
        key_pretrain1, key_pretrain2 = random.split(key_pretrain, 2)
        data_initial = pde_instance.distribution_initial.sample(2048, key_pretrain1)
        data_terminal = pde_instance.distribution_terminal.sample(2048, key_pretrain2)
        data = jnp.concatenate([data_initial, data_terminal], axis=0)
        grad = grad_fn(params, data)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    update = jax.jit(update)

    for key_pretrain in key_pretrains:
        # sample from initial distribution
        params, opt_state = update(key_pretrain, opt_state, params)
    

    return params

        
    
