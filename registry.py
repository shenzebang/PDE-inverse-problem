from example_problems.fokker_planck_example import FokkerPlanck
from example_problems.kinetic_fokker_planck_example import KineticFokkerPlanck
from methods.consistency import ConsistencyBased
from omegaconf import DictConfig

def get_pde_instance(cfg: DictConfig):
    if cfg.pde_instance.name == "Fokker-Planck":
        return FokkerPlanck
    elif cfg.pde_instance.name == "Kinetic-Fokker-Planck":
        return KineticFokkerPlanck
    else:
        return NotImplementedError

def get_method(cfg: DictConfig):
    if cfg.solver.name == "ConsistencyBased":
        return ConsistencyBased
    else:
        raise NotImplementedError
