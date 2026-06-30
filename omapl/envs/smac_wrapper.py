"""SMACv1 / SMACv2 evaluation wrapper.

Adapts the StarCraft Multi-Agent Challenge environments to
:class:`MultiAgentEnv` so trained O-MAPL policies can be evaluated (win rate /
return). Requires an external install NOT pulled in by base requirements::

    pip install git+https://github.com/oxwhirl/smacv2.git
    # plus a StarCraft II installation + the SMAC maps (see scripts/setup_hpc.sh).

SMACv2 specifics
----------------
SMACv2 scenarios are *procedurally generated*: ``StarCraftCapabilityEnvWrapper``
needs a base ``map_name`` (``10gen_{terran,zerg,protoss}``) plus a
``capability_config`` (unit-type distribution + start positions), NOT a name
like ``terran_5_vs_5`` directly. We reconstruct the exact configuration that
OG-MARL / SMACv2 ship (``smacv2/examples/configs/sc2_gen_*.yaml``) so the env's
obs/state/action dimensions match the offline dataset. The scenario string
``<race>_<n_units>_vs_<n_enemies>`` (e.g. ``terran_5_vs_5``, ``terran_10_vs_10``)
is parsed for the unit counts; everything else is taken from the canonical
config (loaded from the *installed* smacv2 package when available, else the
hardcoded fallback below).
"""
from __future__ import annotations

import copy
import os
from typing import Dict, Optional, Tuple

import numpy as np

from .base import MultiAgentEnv


# Canonical SMACv2 env_args, verbatim from smacv2/examples/configs/sc2_gen_*.yaml.
# Only the race-specific bits differ; the rest is shared. n_units / n_enemies are
# overridden per scenario. Used as a fallback when the installed package config
# cannot be read.
_SMACV2_COMMON = dict(
    continuing_episode=False, difficulty="7", game_version=None,
    move_amount=2, obs_all_health=True, obs_instead_of_state=False,
    obs_last_action=False, obs_own_health=True, obs_pathing_grid=False,
    obs_terrain_height=False, obs_timestep_number=False,
    reward_death_value=10, reward_defeat=0, reward_negative_scale=0.5,
    reward_only_positive=True, reward_scale=True, reward_scale_rate=20,
    reward_sparse=False, reward_win=200, replay_dir="", replay_prefix="",
    conic_fov=False, obs_own_pos=True, use_unit_ranges=True,
    min_attack_range=2, num_fov_actions=12, state_last_action=True,
    state_timestep_number=False, step_mul=8, heuristic_ai=False,
    debug=False, prob_obs_enemy=1.0, action_mask=True,
)

_SMACV2_RACE = {
    "terran": dict(
        map_name="10gen_terran",
        unit_types=["marine", "marauder", "medivac"],
        weights=[0.45, 0.45, 0.1], exception_unit_types=["medivac"]),
    "zerg": dict(
        map_name="10gen_zerg",
        unit_types=["zergling", "baneling", "hydralisk"],
        weights=[0.45, 0.1, 0.45], exception_unit_types=["baneling"]),
    "protoss": dict(
        map_name="10gen_protoss",
        unit_types=["stalker", "zealot", "colossus"],
        weights=[0.45, 0.45, 0.1], exception_unit_types=[]),
}


def parse_smacv2_scenario(scenario: str) -> Tuple[str, int, int]:
    """``terran_5_vs_5`` -> ``("terran", 5, 5)``; ``protoss_10_vs_11`` -> (..,10,11)."""
    parts = scenario.split("_")
    if len(parts) != 4 or parts[2] != "vs":
        raise ValueError(
            f"Unrecognised SMACv2 scenario '{scenario}'. Expected "
            f"'<race>_<n_units>_vs_<n_enemies>' (e.g. terran_5_vs_5).")
    race = parts[0]
    if race not in _SMACV2_RACE:
        raise ValueError(f"Unknown SMACv2 race '{race}'.")
    return race, int(parts[1]), int(parts[3])


def _packaged_env_args(race: str) -> Optional[dict]:
    """Load env_args from the installed smacv2 example config (most faithful)."""
    try:
        import smacv2  # type: ignore
        import yaml  # type: ignore
        path = os.path.join(os.path.dirname(smacv2.__file__), "examples",
                            "configs", f"sc2_gen_{race}.yaml")
        with open(path, "r") as f:
            return yaml.safe_load(f)["env_args"]
    except Exception:
        return None


def build_smacv2_env_args(scenario: str) -> dict:
    """Construct the StarCraftCapabilityEnvWrapper kwargs for a scenario."""
    race, n_units, n_enemies = parse_smacv2_scenario(scenario)
    env_args = _packaged_env_args(race)
    if env_args is None:  # hardcoded fallback
        r = _SMACV2_RACE[race]
        team_gen = dict(dist_type="weighted_teams", unit_types=r["unit_types"],
                        weights=r["weights"], observe=True)
        if r["exception_unit_types"]:
            team_gen["exception_unit_types"] = r["exception_unit_types"]
        env_args = dict(_SMACV2_COMMON)
        env_args["map_name"] = r["map_name"]
        env_args["capability_config"] = dict(
            n_units=n_units, n_enemies=n_enemies, team_gen=team_gen,
            start_positions=dict(dist_type="surrounded_and_reflect", p=0.5,
                                 map_x=32, map_y=32))
    else:
        env_args = copy.deepcopy(env_args)
    # Override the unit counts for this specific scenario.
    env_args["capability_config"]["n_units"] = n_units
    env_args["capability_config"]["n_enemies"] = n_enemies
    return env_args


class SMACEnv(MultiAgentEnv):
    def __init__(self, map_name: str, smac_version: int = 2, seed: int = 0,
                 **smac_kwargs):
        self.discrete = True
        try:
            if smac_version == 2:
                from smacv2.env import StarCraftCapabilityEnvWrapper  # type: ignore
                env_args = build_smacv2_env_args(map_name)
                env_args.update(smac_kwargs)
                self._env = StarCraftCapabilityEnvWrapper(seed=seed, **env_args)
            else:
                from smac.env import StarCraft2Env  # type: ignore
                self._env = StarCraft2Env(map_name=map_name, seed=seed, **smac_kwargs)
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "SMAC is not installed. Install smacv2 (and StarCraft II) to "
                "evaluate on SMAC. See this module's docstring / scripts/setup_hpc.sh."
            ) from e

        info = self._env.get_env_info()
        self.n_agents = info["n_agents"]
        self.obs_dim = info["obs_shape"]
        self.state_dim = info["state_shape"]
        self.action_dim = info["n_actions"]

    def reset(self):
        self._env.reset()
        return self._obs()

    def _obs(self):
        obs = np.asarray(self._env.get_obs(), np.float32)            # [n, obs_dim]
        state = np.asarray(self._env.get_state(), np.float32)        # [state_dim]
        avail = np.asarray([self._env.get_avail_agent_actions(i)
                            for i in range(self.n_agents)], np.float32)
        return obs, state, avail

    def step(self, actions: np.ndarray):
        reward, done, info = self._env.step(np.asarray(actions).reshape(-1).tolist())
        obs, state, avail = self._obs()
        out_info = {"won": bool(info.get("battle_won", False))}
        return obs, state, avail, float(reward), bool(done), out_info

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass
