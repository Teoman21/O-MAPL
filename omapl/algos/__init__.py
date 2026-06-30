"""Algorithm registry."""
from .omapl import OMAPL, IPL_VDN
from .baselines import IIPL, BC

ALGOS = {
    "omapl": OMAPL,
    "ipl_vdn": IPL_VDN,
    "iipl": IIPL,
    "bc": BC,
}


def build_algo(cfg):
    if cfg.algo not in ALGOS:
        raise ValueError(f"Unknown algo '{cfg.algo}'. Options: {list(ALGOS)}")
    return ALGOS[cfg.algo](cfg)


__all__ = ["OMAPL", "IPL_VDN", "IIPL", "BC", "ALGOS", "build_algo"]
