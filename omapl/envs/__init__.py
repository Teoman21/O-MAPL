from .base import MultiAgentEnv
from .synthetic import CoordinationEnv


def build_env(cfg):
    """Construct an evaluation env from a Config (lazy imports for heavy deps)."""
    if cfg.env == "synthetic":
        return CoordinationEnv(
            n_agents=cfg.n_agents or 3,
            action_dim=cfg.action_dim or 4,
            episode_len=int(cfg.extra.get("episode_len", 20)),
            seed=cfg.seed,
        )
    if cfg.env == "smacv2":
        from .smac_wrapper import SMACEnv
        return SMACEnv(map_name=cfg.task, smac_version=2, seed=cfg.seed)
    if cfg.env == "smacv1":
        from .smac_wrapper import SMACEnv
        return SMACEnv(map_name=cfg.task, smac_version=1, seed=cfg.seed)
    if cfg.env == "mamujoco":
        from .mamujoco_wrapper import MAMuJoCoEnv
        return MAMuJoCoEnv(scenario=cfg.extra.get("scenario", cfg.task),
                           agent_conf=cfg.extra.get("agent_conf", ""), seed=cfg.seed)
    raise ValueError(f"Unknown env '{cfg.env}'")


__all__ = ["MultiAgentEnv", "CoordinationEnv", "build_env"]
