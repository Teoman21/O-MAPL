"""Adapter to turn OMIGA / ComaDICE offline datasets into trajectory lists.

The paper sources its offline data from OMIGA (Wang et al., 2022; SMACv1 &
MAMuJoCo) and ComaDICE (Bui et al., 2025; SMACv2). Those releases store
per-task transition buffers (typically HDF5 / npy) with arrays such as
``obs``, ``actions``, ``rewards``, ``next_obs``, ``state``, ``terminals`` and
(for SMAC) ``available_actions``.

Because the exact on-disk layout differs per release, this module exposes a
*format-agnostic core* (:func:`trajectories_from_arrays`) plus thin loaders you
can adapt. Point ``--data_path`` at the produced PreferenceDataset after running
``omapl.data.generate_preferences`` on the trajectories returned here.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .preference_dataset import DataSpec


def trajectories_from_arrays(
    obs: np.ndarray,            # [N, n_agents, obs_dim]
    actions: np.ndarray,        # [N, n_agents] (discrete) or [N, n_agents, act]
    next_obs: np.ndarray,       # [N, n_agents, obs_dim]
    state: np.ndarray,          # [N, state_dim]
    next_state: np.ndarray,     # [N, state_dim]
    terminals: np.ndarray,      # [N] 1.0 at the last transition of an episode
    discrete: bool,
    avail: Optional[np.ndarray] = None,        # [N, n_agents, n_actions]
    next_avail: Optional[np.ndarray] = None,
    rewards: Optional[np.ndarray] = None,      # [N] (only for quality ranking)
) -> Tuple[List[Dict[str, np.ndarray]], DataSpec]:
    """Slice flat transition arrays into per-episode trajectory dicts."""
    terminals = np.asarray(terminals).reshape(-1).astype(bool)
    bounds = np.where(terminals)[0]
    starts = np.concatenate([[0], bounds[:-1] + 1])
    ends = bounds + 1  # exclusive

    trajs: List[Dict[str, np.ndarray]] = []
    for s, e in zip(starts, ends):
        tr = {
            "obs": obs[s:e].astype(np.float32),
            "actions": actions[s:e].astype(np.float32),
            "next_obs": next_obs[s:e].astype(np.float32),
            "state": state[s:e].astype(np.float32),
            "next_state": next_state[s:e].astype(np.float32),
            "done": terminals[s:e].astype(np.float32),
        }
        if avail is not None:
            tr["avail"] = avail[s:e].astype(np.float32)
        if next_avail is not None:
            tr["next_avail"] = next_avail[s:e].astype(np.float32)
        if rewards is not None:
            tr["return"] = float(np.asarray(rewards[s:e]).sum())
        trajs.append(tr)

    n_agents = obs.shape[1]
    obs_dim = obs.shape[2]
    state_dim = state.shape[1]
    if discrete:
        action_dim = int(avail.shape[-1]) if avail is not None else int(actions.max() + 1)
    else:
        action_dim = actions.shape[-1]
    spec = DataSpec(n_agents, obs_dim, state_dim, action_dim, discrete)
    return trajs, spec


def load_hdf5(path: str, key_map: Optional[Dict[str, str]] = None):
    """Load an OMIGA/ComaDICE-style HDF5 buffer into the standard arrays.

    ``key_map`` overrides dataset key names if a release differs from the
    defaults below. Requires ``h5py``.
    """
    import h5py  # type: ignore
    km = {
        "obs": "obs", "actions": "actions", "next_obs": "next_obs",
        "state": "state", "next_state": "next_state",
        "terminals": "terminals", "avail": "available_actions",
        "next_avail": "next_available_actions", "rewards": "rewards",
    }
    if key_map:
        km.update(key_map)
    with h5py.File(path, "r") as f:
        def get(k):
            return np.asarray(f[km[k]]) if km[k] in f else None
        return {
            "obs": get("obs"), "actions": get("actions"),
            "next_obs": get("next_obs"), "state": get("state"),
            "next_state": get("next_state"), "terminals": get("terminals"),
            "avail": get("avail"), "next_avail": get("next_avail"),
            "rewards": get("rewards"),
        }
