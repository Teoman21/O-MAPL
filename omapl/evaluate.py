"""Evaluate decentralised policies in an environment (returns / win rate)."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


@torch.no_grad()
def evaluate_policy(algo, env, n_episodes: int = 32, device="cpu",
                    deterministic: bool = True) -> Dict[str, float]:
    """Run ``n_episodes`` and report mean/std return and (if available) win rate.

    Execution is fully decentralised: each agent acts from its local
    observation o_i via pi_i(.|o_i), exactly as in CTDE deployment.
    """
    returns, wins = [], []
    for _ in range(n_episodes):
        obs, state, avail = env.reset()
        ep_ret, done, won = 0.0, False, None
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            avail_t = (torch.as_tensor(avail, dtype=torch.float32, device=device)
                       if avail is not None else None)
            actions = algo.act(obs_t, avail_t, deterministic=deterministic)
            actions = actions.cpu().numpy()
            obs, state, avail, reward, done, info = env.step(actions)
            ep_ret += reward
            if done and "won" in info:
                won = bool(info["won"])
        returns.append(ep_ret)
        if won is not None:
            wins.append(float(won))
    out = {"return_mean": float(np.mean(returns)),
           "return_std": float(np.std(returns))}
    if wins:
        out["win_rate"] = float(np.mean(wins)) * 100.0  # percent, as in the paper
    return out
