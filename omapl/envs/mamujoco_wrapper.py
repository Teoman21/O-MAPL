"""MAMuJoCo wrapper (WIRING POINT) for continuous-control evaluation.

Requires MuJoCo and the multi-agent MuJoCo package, which are NOT installed by
this repo's base requirements::

    pip install mujoco gymnasium
    pip install git+https://github.com/schroederdewitt/multiagent_mujoco.git

The paper evaluates Hopper-v2, Ant-v2, HalfCheetah-v2 with the standard agent
factorisations from de Witt et al. (2020). The offline datasets come from
OMIGA (HAPPO-trained, medium-replay/medium/expert qualities).
"""
from __future__ import annotations

import numpy as np

from .base import MultiAgentEnv


class MAMuJoCoEnv(MultiAgentEnv):
    def __init__(self, scenario: str, agent_conf: str, seed: int = 0, **kwargs):
        self.discrete = False
        try:
            from multiagent_mujoco.mujoco_multi import MujocoMulti  # type: ignore
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "multiagent_mujoco is not installed. See this module's docstring."
            ) from e
        self._env = MujocoMulti(env_args={
            "scenario": scenario, "agent_conf": agent_conf,
            "agent_obsk": kwargs.get("agent_obsk", 1), "seed": seed})
        info = self._env.get_env_info()
        self.n_agents = info["n_agents"]
        self.obs_dim = info["obs_shape"]
        self.state_dim = info["state_shape"]
        self.action_dim = info["n_actions"]   # per-agent continuous action dim

    def reset(self):
        self._env.reset()
        return self._obs()

    def _obs(self):
        obs = np.asarray(self._env.get_obs(), np.float32)
        state = np.asarray(self._env.get_state(), np.float32)
        return obs, state, None

    def step(self, actions: np.ndarray):
        reward, done, info = self._env.step(np.asarray(actions, np.float32))
        obs, state, _ = self._obs()
        return obs, state, None, float(reward), bool(done), dict(info)
