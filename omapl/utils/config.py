"""Configuration handling for O-MAPL.

A light-weight, dependency-free config layer: a dataclass with sensible
defaults (matching Table 6 of the paper) that can be overridden from a YAML
file and/or command-line ``key=value`` pairs.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:  # pragma: no cover - yaml is optional for parsing configs
    _HAS_YAML = False


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # Experiment / bookkeeping
    # ------------------------------------------------------------------ #
    algo: str = "omapl"               # one of: omapl, ipl_vdn, iipl, bc
    env: str = "synthetic"            # synthetic | smacv2 | mamujoco
    task: str = "synthetic-coop"      # task name (e.g. protoss_5_vs_5)
    data_path: str = ""               # path to preference dataset (.npz/.pkl)
    seed: int = 0
    device: str = "cpu"               # "cuda" if available
    log_dir: str = "runs"
    exp_name: str = "omapl"

    # ------------------------------------------------------------------ #
    # Problem dimensions (filled in from the dataset/env if left as None)
    # ------------------------------------------------------------------ #
    n_agents: int = 0
    obs_dim: int = 0
    state_dim: int = 0
    action_dim: int = 0               # n discrete actions, or continuous dim
    discrete: bool = True

    # ------------------------------------------------------------------ #
    # Optimisation (Table 6)
    # ------------------------------------------------------------------ #
    lr: float = 1e-4                  # learning rate (Q-value & policy nets)
    tau: float = 0.005                # soft target update rate
    gamma: float = 0.99               # discount factor
    batch_size: int = 32              # number of preference *pairs* per batch
    agent_hidden_dim: int = 256       # local q/v/policy hidden dim
    mixer_hidden_dim: int = 64        # hypernetwork hidden dim
    n_train_steps: int = 100_000

    # ------------------------------------------------------------------ #
    # O-MAPL specific
    # ------------------------------------------------------------------ #
    beta: float = 1.0                 # MaxEnt temperature (not given in paper)
    reg_coef: float = 1.0             # weight of the chi^2 regulariser phi(.)
    param_sharing: bool = True        # share local network params across agents
    use_agent_id: bool = True         # append a one-hot agent id to local input
    mixer_q_use_action: bool = True   # condition the Q-mixer on the joint action
    exp_clip: float = 8.0             # clamp on (Q-V)/beta exponent for stability
    adv_weight_clip: float = 100.0    # clamp on the WBC advantage weight
    grad_clip: float = 10.0           # global grad-norm clip (0 to disable)

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    eval_every: int = 1000
    eval_episodes: int = 32           # episodes per evaluation step (Table 6)
    n_eval_steps: int = 100           # number of evaluation checkpoints (Table 6)

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        if not _HAS_YAML:
            raise RuntimeError("PyYAML is required to read YAML configs.")
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        known = {f.name for f in dataclasses.fields(cls)}
        kwargs, extra = {}, {}
        for k, v in data.items():
            (kwargs if k in known else extra)[k] = v
        cfg = cls(**kwargs)
        cfg.extra.update(extra)
        return cfg

    def update_from_overrides(self, overrides: list[str]) -> "Config":
        """Apply ``key=value`` CLI overrides, coercing to the field type."""
        known = {f.name: f.type for f in dataclasses.fields(self)}
        for ov in overrides:
            if "=" not in ov:
                continue
            k, v = ov.split("=", 1)
            if k in known:
                setattr(self, k, _coerce(v, getattr(self, k)))
            else:
                self.extra[k] = v
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _coerce(value: str, reference: Any) -> Any:
    """Coerce a string CLI value to the type of the existing field value."""
    if isinstance(reference, bool):
        return value.lower() in ("1", "true", "yes", "y")
    if isinstance(reference, int):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value
