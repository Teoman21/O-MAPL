"""Generate a synthetic rule-based preference dataset for the CoordinationEnv.

This produces a small, fully self-contained PreferenceDataset so the whole
O-MAPL pipeline can be run and validated without StarCraft/MuJoCo or any large
downloads. Behaviour policies of three qualities (poor / medium / expert) are
rolled out, then trajectory pairs are labelled by quality (rule-based, IPL-style).

    python scripts/make_synthetic_data.py --out data/synthetic.pkl
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omapl.data.preference_dataset import DataSpec
from omapl.data.generate_preferences import (
    collect_trajectories, synthetic_behavior_policy, build_preference_dataset)
from omapl.envs.synthetic import CoordinationEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data/synthetic.pkl")
    p.add_argument("--n_agents", type=int, default=3)
    p.add_argument("--action_dim", type=int, default=4)
    p.add_argument("--episode_len", type=int, default=20)
    p.add_argument("--n_trajs_per_quality", type=int, default=200)
    p.add_argument("--n_pairs", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    env = CoordinationEnv(n_agents=args.n_agents, action_dim=args.action_dim,
                          episode_len=args.episode_len, seed=args.seed)

    qualities = {"poor": 0.8, "medium": 0.4, "expert": 0.05}
    trajs_by_quality = {}
    for name, eps in qualities.items():
        trajs, mean_ret = collect_trajectories(
            env, synthetic_behavior_policy(eps), args.n_trajs_per_quality, rng)
        trajs_by_quality[name] = trajs
        print(f"  {name:7s} (eps={eps}): mean return {mean_ret:.2f} "
              f"over {len(trajs)} trajs")

    spec = DataSpec(args.n_agents, env.obs_dim, env.state_dim,
                    args.action_dim, discrete=True)
    ds = build_preference_dataset(
        trajs_by_quality, ["poor", "medium", "expert"], args.n_pairs, spec, rng=rng)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ds.save(args.out)
    print(f"Saved {len(ds)} preference pairs -> {args.out}")


if __name__ == "__main__":
    main()
