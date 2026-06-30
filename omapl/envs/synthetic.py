"""A small synthetic *cooperative* multi-agent env for end-to-end validation.

`CoordinationEnv` is a fully cooperative signalling game:

* At each step a global discrete *signal* ``s in {0, ..., A-1}`` is broadcast.
  Every agent observes a (noisy) one-hot encoding of the signal.
* The reward is the fraction of agents whose action matches the signal, with a
  coordination bonus when *all* agents match. The unique optimal decentralised
  policy is "play the signal you observe".

This env has a known optimum, so it lets us verify that O-MAPL actually
*learns* (returns should climb from the random ~1/A level toward the maximum)
without needing StarCraft II or MuJoCo. It also serves as the behaviour-policy
source for generating synthetic preference data (see
``omapl.data.generate_preferences``).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .base import MultiAgentEnv


class CoordinationEnv(MultiAgentEnv):
    def __init__(self, n_agents: int = 3, action_dim: int = 4,
                 episode_len: int = 20, obs_noise: float = 0.1,
                 win_threshold: float = 0.9, seed: int = 0):
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.discrete = True
        self.episode_len = episode_len
        self.obs_noise = obs_noise
        self.win_threshold = win_threshold
        self.obs_dim = action_dim                  # one-hot of the signal
        self.state_dim = action_dim                # global one-hot signal
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._signal = 0
        self._reward_sum = 0.0

    # ------------------------------------------------------------------ #
    def _obs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        onehot = np.zeros(self.action_dim, np.float32)
        onehot[self._signal] = 1.0
        noise = self._rng.normal(0, self.obs_noise, size=(self.n_agents, self.action_dim))
        obs = (onehot[None, :] + noise).astype(np.float32)
        state = onehot.copy()
        avail = np.ones((self.n_agents, self.action_dim), np.float32)
        return obs, state, avail

    def reset(self):
        self._t = 0
        self._reward_sum = 0.0
        self._signal = int(self._rng.integers(self.action_dim))
        return self._obs()

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions).reshape(-1)
        n_match = int((actions == self._signal).sum())
        reward = n_match / self.n_agents
        if n_match == self.n_agents:
            reward += 0.0  # (kept explicit; bonus tunable)
        self._reward_sum += reward
        self._t += 1
        done = self._t >= self.episode_len
        # Advance the signal for the next step.
        self._signal = int(self._rng.integers(self.action_dim))
        obs, state, avail = self._obs()
        info = {}
        if done:
            info["won"] = (self._reward_sum / self.episode_len) >= self.win_threshold
            info["return"] = self._reward_sum
        return obs, state, avail, reward, done, info
