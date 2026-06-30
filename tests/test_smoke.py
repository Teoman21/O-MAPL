"""End-to-end smoke test on the synthetic CoordinationEnv.

Generates a tiny rule-based preference dataset, trains O-MAPL for a short while,
and checks that the learned decentralised policy clearly beats a random policy
(the env's optimum is "play the observed signal").
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omapl.algos import build_algo
from omapl.data.preference_dataset import DataSpec
from omapl.data.generate_preferences import (
    collect_trajectories, synthetic_behavior_policy, build_preference_dataset)
from omapl.envs.synthetic import CoordinationEnv
from omapl.evaluate import evaluate_policy
from omapl.utils.config import Config
from omapl.utils.torch_utils import set_seed


def _make_dataset(n_agents=3, action_dim=4, episode_len=10, seed=0):
    rng = np.random.default_rng(seed)
    env = CoordinationEnv(n_agents, action_dim, episode_len, seed=seed)
    trajs_by_quality = {}
    for name, eps in {"poor": 0.8, "medium": 0.4, "expert": 0.05}.items():
        trajs, _ = collect_trajectories(env, synthetic_behavior_policy(eps), 60, rng)
        trajs_by_quality[name] = trajs
    spec = DataSpec(n_agents, env.obs_dim, env.state_dim, action_dim, True)
    ds = build_preference_dataset(trajs_by_quality,
                                  ["poor", "medium", "expert"], 500, spec, rng=rng)
    return ds, spec


def test_omapl_learns(algo_name="omapl", n_steps=1500):
    set_seed(0)
    n_agents, action_dim, episode_len = 3, 4, 10
    ds, spec = _make_dataset(n_agents, action_dim, episode_len)
    cfg = Config(algo=algo_name, env="synthetic", n_agents=n_agents,
                 obs_dim=spec.obs_dim, state_dim=spec.state_dim,
                 action_dim=action_dim, discrete=True, agent_hidden_dim=64,
                 mixer_hidden_dim=32, batch_size=32, beta=1.0, device="cpu")
    algo = build_algo(cfg)
    env = CoordinationEnv(n_agents, action_dim, episode_len, seed=123)

    random_score = evaluate_policy(algo, env, n_episodes=20)["return_mean"]
    rng = np.random.default_rng(0)
    for _ in range(n_steps):
        algo.update(ds.sample_batch(cfg.batch_size, rng=rng))
    trained_score = evaluate_policy(algo, env, n_episodes=50)["return_mean"]

    chance = episode_len / action_dim  # ~2.5: every agent matches w.p. 1/A
    print(f"[{algo_name}] random={random_score:.2f}  trained={trained_score:.2f}  "
          f"chance≈{chance:.2f}  optimum={episode_len}")
    assert trained_score > random_score + 1.0, (random_score, trained_score)
    assert trained_score > chance * 1.3, (trained_score, chance)
    return trained_score


if __name__ == "__main__":
    score = test_omapl_learns("omapl")
    print(f"\nSmoke test passed (O-MAPL trained return = {score:.2f}).")
