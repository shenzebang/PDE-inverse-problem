from example_problems.fokker_planck_example import FokkerPlanck
from example_problems.kinetic_fokker_planck_example_OU import KineticFokkerPlanck as KFPOU
from example_problems.kinetic_fokker_planck_example_GMM import KineticFokkerPlanck as KFPGMM
from example_problems.kinetic_mckean_vlasov_example_quadratic import KineticMcKeanVlasov as KMVOU
from methods.consistency import ConsistencyBased
from omegaconf import DictConfig

KineticFokkerPlanckPotential = {
    "Quadratic": KFPOU,
    "GMM": KFPGMM,
}

KineticMcKeanVlasovPotential = {
    "Quadratic": KMVOU,
}


def get_pde_instance(cfg: DictConfig):
    if cfg.pde_instance.name == "Fokker-Planck":
        return FokkerPlanck
    elif cfg.pde_instance.name == "Kinetic-Fokker-Planck":
        return KineticFokkerPlanckPotential[cfg.pde_instance.potential]
    elif cfg.pde_instance.name == "Kinetic-McKean-Vlasov":
        return KineticMcKeanVlasovPotential[cfg.pde_instance.potential]
    else:
        return NotImplementedError

def get_method(cfg: DictConfig):
    if cfg.solver.name == "ConsistencyBased":
        return ConsistencyBased
    else:
        raise NotImplementedError
