from example_problems.fokker_planck_example import FokkerPlanck
from methods.consistency import ConsistencyBased
from omegaconf import DictConfig

def get_pde_instance(cfg: DictConfig):
    if cfg.pde_instance.name == "Fokker-Planck":
        return FokkerPlanck
    else:
        return NotImplementedError

def get_method(cfg: DictConfig):
    if cfg.solver.name == "ConsistencyBased":
        return ConsistencyBased
    else:
        raise NotImplementedError
