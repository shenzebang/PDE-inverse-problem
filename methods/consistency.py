import methods.consistency_instances.fokker_planck as fokker_planck
import methods.consistency_instances.kinetic_fokker_planck as kinetic_fokker_planck
from api import Method
from functools import partial
import jax.random as random

INSTANCES={
    "Fokker-Planck": fokker_planck,
    "Kinetic-Fokker-Planck": kinetic_fokker_planck,
}


class ConsistencyBased(Method):
    def create_model_fn(self):
        if self.cfg.pde_instance.name in INSTANCES:
            return INSTANCES[self.cfg.pde_instance.name].create_model_fn(self.pde_instance)
        else:
            raise NotImplementedError


    def test_fn(self, forward_fn, params, rng):
        forward_fn = partial(forward_fn, params)
        if self.cfg.pde_instance.name in INSTANCES:
            return INSTANCES[self.cfg.pde_instance.name].test_fn(forward_fn=forward_fn, pde_instance=self.pde_instance,
                                                rng=rng)
        else:
            raise NotImplementedError

    def value_and_grad_fn(self, forward_fn, params, rng):
        rng_sample, rng_vg = random.split(rng, 2)
        # Sample data
        data = self.sample_data(rng_sample)
        # compute function value and gradient
        if self.cfg.pde_instance.name in INSTANCES:
            return INSTANCES[self.cfg.pde_instance.name].value_and_grad_fn(forward_fn=forward_fn, 
                                                                           params=params, data=data, 
                                                                           rng=rng_vg, pde_instance=self.pde_instance)
        else:
            raise NotImplementedError

    def sample_data(self, rng):
        rng_initial, rng_terminal, rng_0T = random.split(rng, 3)
        data = {
            "initial"  : self.pde_instance.distribution_initial.sample(self.cfg.train.batch_size, rng_initial),
            "terminal" : self.pde_instance.distribution_terminal.sample(self.cfg.train.batch_size, rng_terminal),
            "0T"       : self.pde_instance.sample_ground_truth(rng_0T, self.cfg.train.batch_size)
        }
        # For the Kinetic models, [x, v] are concatenated. 
        # The instance should handle this on its own.
        return data