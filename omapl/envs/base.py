"""Common multi-agent environment interface used for evaluation.

Training is fully offline (from a :class:`PreferenceDataset`); environments are
only needed to *evaluate* the learned decentralised policies. Any env that
implements this interface can be plugged into ``omapl.evaluate``.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


class MultiAgentEnv:
    n_agents: int
    obs_dim: int
    state_dim: int
    action_dim: int
    discrete: bool

    def reset(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Return ``(obs [n, obs_dim], state [state_dim], avail [n, A] or None)``."""
        raise NotImplementedError

    def step(self, actions: np.ndarray):
        """Return ``(obs, state, avail, reward: float, done: bool, info: dict)``.

        ``info`` may contain ``"won"`` (bool) for competitive envs like SMAC.
        """
        raise NotImplementedError
