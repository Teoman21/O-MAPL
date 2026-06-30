"""Offline pairwise-preference dataset for multi-agent PbRL.

Data model (matches the paper's setup, Section 4.1 / Table 3):

* A *trajectory* sigma is a sequence of joint transitions
  ``(o, a, o')`` plus the global ``state``/``next_state`` and (for discrete
  envs) available-action masks. Trajectories are stored **once**.
* A *preference* is a pair ``(sigma1, sigma2)`` with a label indicating which
  is preferred. We store pairs as index references into the trajectory list so
  the large SMAC/MAMuJoCo offline datasets are not duplicated.

Label convention: ``label`` is the probability that ``sigma1`` (the first
element of the pair) is preferred — 1.0 (first preferred), 0.0 (second
preferred), 0.5 (tie). This supports the Bradley-Terry soft cross-entropy used
by the preference loss and the LLM "#0" tie outputs.

Per-trajectory arrays (numpy), shapes with T = trajectory length:
    obs         [T, n_agents, obs_dim]
    actions     [T, n_agents]                 (discrete indices)
                [T, n_agents, action_dim]     (continuous)
    next_obs    [T, n_agents, obs_dim]
    state       [T, state_dim]
    next_state  [T, state_dim]
    avail       [T, n_agents, action_dim]      (discrete; optional)
    next_avail  [T, n_agents, action_dim]      (discrete; optional)
    done        [T]                            (1.0 at terminal transition)
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class DataSpec:
    n_agents: int
    obs_dim: int
    state_dim: int
    action_dim: int
    discrete: bool


class PreferenceDataset:
    def __init__(self, trajectories: List[Dict[str, np.ndarray]],
                 pairs: np.ndarray, labels: np.ndarray, spec: DataSpec):
        self.trajectories = trajectories
        self.pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        self.labels = np.asarray(labels, dtype=np.float32).reshape(-1)
        self.spec = spec
        assert len(self.pairs) == len(self.labels)

    def __len__(self) -> int:
        return len(self.pairs)

    # ------------------------------------------------------------------ #
    def sample_batch(self, batch_size: int, device="cpu",
                     rng: Optional[np.random.Generator] = None) -> Dict[str, torch.Tensor]:
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, len(self.pairs), size=batch_size)
        return self._collate(idx, device)

    def iter_batches(self, batch_size: int, device="cpu"):
        for start in range(0, len(self.pairs), batch_size):
            idx = np.arange(start, min(start + batch_size, len(self.pairs)))
            yield self._collate(idx, device)

    # ------------------------------------------------------------------ #
    def _collate(self, pair_idx: np.ndarray, device) -> Dict[str, torch.Tensor]:
        """Gather a batch of pairs into padded tensors ``[B, 2, T, ...]``."""
        B = len(pair_idx)
        spec = self.spec
        # Determine the max length across the 2B trajectories in this batch.
        lengths = np.array(
            [[len(self.trajectories[self.pairs[p, k]]["actions"]) for k in (0, 1)]
             for p in pair_idx]
        )
        T = int(lengths.max())
        n, od, sd, ad = spec.n_agents, spec.obs_dim, spec.state_dim, spec.action_dim

        act_shape = (B, 2, T, n) if spec.discrete else (B, 2, T, n, ad)
        out = {
            "obs": np.zeros((B, 2, T, n, od), np.float32),
            "next_obs": np.zeros((B, 2, T, n, od), np.float32),
            "actions": np.zeros(act_shape, np.float32),
            "state": np.zeros((B, 2, T, sd), np.float32),
            "next_state": np.zeros((B, 2, T, sd), np.float32),
            "avail": np.ones((B, 2, T, n, ad), np.float32) if spec.discrete else None,
            "next_avail": np.ones((B, 2, T, n, ad), np.float32) if spec.discrete else None,
            "done": np.zeros((B, 2, T), np.float32),
            "mask": np.zeros((B, 2, T), np.float32),
        }
        for b, p in enumerate(pair_idx):
            for k in (0, 1):
                tr = self.trajectories[self.pairs[p, k]]
                t = len(tr["actions"])
                out["obs"][b, k, :t] = tr["obs"]
                out["next_obs"][b, k, :t] = tr["next_obs"]
                out["actions"][b, k, :t] = tr["actions"]
                out["state"][b, k, :t] = tr["state"]
                out["next_state"][b, k, :t] = tr["next_state"]
                out["done"][b, k, :t] = tr.get("done", np.zeros(t, np.float32))
                out["mask"][b, k, :t] = 1.0
                if spec.discrete:
                    if "avail" in tr:
                        out["avail"][b, k, :t] = tr["avail"]
                    if "next_avail" in tr:
                        out["next_avail"][b, k, :t] = tr["next_avail"]

        batch = {k: torch.as_tensor(v, device=device)
                 for k, v in out.items() if v is not None}
        batch["label"] = torch.as_tensor(self.labels[pair_idx], device=device)
        return batch

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {"trajectories": self.trajectories, "pairs": self.pairs,
                 "labels": self.labels, "spec": self.spec},
                f, protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str) -> "PreferenceDataset":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["trajectories"], d["pairs"], d["labels"], d["spec"])
