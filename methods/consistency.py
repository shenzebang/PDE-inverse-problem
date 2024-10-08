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
        if self.pde_instance.sample_scheme == "exact":
            batch_size_0T = {
                "random_time": self.cfg.solver.train.batch_size_0T,
                "grid_time": (self.cfg.solver.train.n_time_stamps, self.cfg.solver.train.sample_per_time)
            }
            data = {
                "initial"  : self.pde_instance.distribution_initial.sample(self.cfg.solver.train.batch_size_init, rng_initial),
                "terminal" : self.pde_instance.distribution_terminal.sample(self.cfg.solver.train.batch_size_terminal, rng_terminal),
                "0T"       : self.pde_instance.sample_ground_truth(rng_0T, batch_size_0T[self.cfg.solver.train.sample_mode])
            }
        elif self.pde_instance.sample_scheme == "SDE":
            # always use grid_time mode in the SDE sampling scheme
            data = {}
            # data = {
            #     "initial"  : self.pde_instance.distribution_initial.sample(self.cfg.solver.train.batch_size_init, rng_initial),
            # }
            # sample ground truth should return the initial, terminal and 0T samples
            data["initial"], data["terminal"], data["0T"] = self.pde_instance.sample_ground_truth(rng_0T, self.cfg.solver.train.batch_size_0T)
        else:
            raise ValueError("unknown sampling scheme")
        # For the Kinetic models, [x, v] are concatenated. 
        # The instance should handle this on its own.
        return data