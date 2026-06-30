"""Generate offline pairwise-preference datasets (Section 6 / Appendix B.1-B.2).

Two labelling methods from the paper:

* **Rule-based** (following IPL): sample trajectory pairs from offline datasets
  of varying quality (poor / medium / expert) and assign a binary preference
  label based on dataset quality (the higher-quality trajectory is preferred).
* **LLM-based** (following DPM): build a prompt from the final-state summary of
  each trajectory (health, deaths, remaining health, length) and ask an LLM
  (e.g. GPT-4o) which trajectory is better. The exact SMAC prompt template
  (Table 5 of the paper) is reproduced in :func:`build_smac_preference_prompt`.

This module is environment-agnostic: it operates on lists of trajectory dicts
(see :mod:`omapl.data.preference_dataset`). Helpers to collect such trajectories
by rolling out behaviour policies are also provided (used for the synthetic env
and reusable for any :class:`MultiAgentEnv`).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .preference_dataset import DataSpec, PreferenceDataset


# ---------------------------------------------------------------------------- #
# Trajectory collection (for the synthetic env / any MultiAgentEnv)
# ---------------------------------------------------------------------------- #
def collect_trajectories(env, policy_fn: Callable, n_trajs: int,
                         rng: Optional[np.random.Generator] = None
                         ) -> Tuple[List[Dict[str, np.ndarray]], float]:
    """Roll out ``policy_fn(obs, state, avail, rng) -> actions`` for n episodes.

    Returns the list of trajectory dicts and the mean episodic return.
    """
    rng = rng or np.random.default_rng()
    trajs, returns = [], []
    for _ in range(n_trajs):
        obs, state, avail = env.reset()
        buf = {k: [] for k in ("obs", "actions", "next_obs", "state",
                               "next_state", "avail", "next_avail", "done")}
        ep_ret, done = 0.0, False
        while not done:
            act = policy_fn(obs, state, avail, rng)
            nobs, nstate, navail, reward, done, info = env.step(act)
            buf["obs"].append(obs); buf["actions"].append(act)
            buf["next_obs"].append(nobs); buf["state"].append(state)
            buf["next_state"].append(nstate)
            buf["avail"].append(avail if avail is not None else np.ones_like(act))
            buf["next_avail"].append(navail if navail is not None else np.ones_like(act))
            buf["done"].append(float(done))
            obs, state, avail = nobs, nstate, navail
            ep_ret += reward
        traj = {
            "obs": np.asarray(buf["obs"], np.float32),
            "actions": np.asarray(buf["actions"], np.float32),
            "next_obs": np.asarray(buf["next_obs"], np.float32),
            "state": np.asarray(buf["state"], np.float32),
            "next_state": np.asarray(buf["next_state"], np.float32),
            "avail": np.asarray(buf["avail"], np.float32),
            "next_avail": np.asarray(buf["next_avail"], np.float32),
            "done": np.asarray(buf["done"], np.float32),
            "return": ep_ret,
        }
        trajs.append(traj)
        returns.append(ep_ret)
    return trajs, float(np.mean(returns))


def synthetic_behavior_policy(epsilon: float):
    """Behaviour policy for ``CoordinationEnv`` with quality controlled by eps.

    With prob (1 - eps) it plays the (noisily observed) signal; otherwise random.
    eps=0 is expert, eps=1 is fully random.
    """
    def policy_fn(obs, state, avail, rng):
        n, A = obs.shape
        greedy = obs.argmax(axis=-1)
        rand = rng.integers(0, A, size=n)
        explore = rng.random(n) < epsilon
        return np.where(explore, rand, greedy)
    return policy_fn


# ---------------------------------------------------------------------------- #
# Rule-based preference labelling
# ---------------------------------------------------------------------------- #
def make_rule_based_pairs(trajs_by_quality: "dict[str, list]",
                          quality_order: Sequence[str], n_pairs: int,
                          rng: Optional[np.random.Generator] = None,
                          same_tier_by_return: bool = True
                          ) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """Build (flat trajectory list, pairs, labels) from quality-bucketed trajs.

    ``quality_order`` lists tiers from worst to best (e.g. ``["poor","medium",
    "expert"]``). A pair drawn from two different tiers is labelled by tier (the
    better tier is preferred). Same-tier pairs are labelled by return (or tie).
    """
    rng = rng or np.random.default_rng()
    flat: List[Dict] = []
    index_by_tier: Dict[str, List[int]] = {}
    for tier in quality_order:
        idxs = []
        for tr in trajs_by_quality.get(tier, []):
            idxs.append(len(flat)); flat.append(tr)
        index_by_tier[tier] = idxs
    rank = {t: r for r, t in enumerate(quality_order)}

    pairs, labels = [], []
    for _ in range(n_pairs):
        ta, tb = rng.choice(quality_order, size=2, replace=True)
        ia = int(rng.choice(index_by_tier[ta]))
        ib = int(rng.choice(index_by_tier[tb]))
        if rank[ta] != rank[tb]:
            label = 1.0 if rank[ta] > rank[tb] else 0.0
        elif same_tier_by_return:
            ra, rb = flat[ia].get("return", 0.0), flat[ib].get("return", 0.0)
            label = 1.0 if ra > rb else (0.0 if rb > ra else 0.5)
        else:
            label = 0.5
        pairs.append([ia, ib]); labels.append(label)
    return flat, np.asarray(pairs, np.int64), np.asarray(labels, np.float32)


def build_preference_dataset(trajs_by_quality, quality_order, n_pairs, spec,
                             rng=None, **kw) -> PreferenceDataset:
    flat, pairs, labels = make_rule_based_pairs(
        trajs_by_quality, quality_order, n_pairs, rng=rng, **kw)
    return PreferenceDataset(flat, pairs, labels, spec)


# ---------------------------------------------------------------------------- #
# LLM-based preference labelling (Table 5)
# ---------------------------------------------------------------------------- #
def build_smac_preference_prompt(scenario: str, ally_config: str,
                                 enemy_config: str, traj1: Dict, traj2: Dict,
                                 objective: Optional[str] = None) -> str:
    """Reproduce the SMAC preference prompt of Table 5 (paper, Appendix B.2).

    ``traj{1,2}`` are dicts with keys: ``ally_health`` (list[float]),
    ``enemy_health`` (list[float]), ``n_ally_deaths`` (int),
    ``n_enemy_deaths`` (int), ``total_ally_health`` (float),
    ``total_enemy_health`` (float), ``n_steps`` (int).
    """
    objective = objective or (
        "Defeat all enemy agents while ensuring as many allied agents as "
        "possible survive.")

    def fmt(tr: Dict, idx: int) -> str:
        ah = ", ".join(f"{h:.3f}" for h in tr["ally_health"])
        eh = ", ".join(f"{h:.3f}" for h in tr["enemy_health"])
        return (
            f"[Trajectory {idx}]\n"
            f"1. Final State Information\n"
            f"1) Allied Agents Health : {ah}\n"
            f"2) Enemy Agents Health : {eh}\n"
            f"3) Number of Allied Deaths : {tr['n_ally_deaths']}\n"
            f"4) Number of Enemy Deaths : {tr['n_enemy_deaths']}\n"
            f"5) Total Remaining Health of Allies : {tr['total_ally_health']:.3f}\n"
            f"6) Total Remaining Health of Enemies : {tr['total_enemy_health']:.3f}\n"
            f"2. Total Number of Steps : {tr['n_steps']}\n")

    return (
        "You are a helpful and honest judge of good game playing and progress in "
        "the StarCraft Multi-Agent Challenge game. Always answer as helpfully as "
        "possible, while being truthful.\n"
        "If you don't know the answer to a question, please don't share false "
        "information.\n"
        "I'm looking to have you evaluate a scenario in the StarCraft Multi-Agent "
        "Challenge. Your role will be to assess how much the actions taken by "
        "multiple agents in a given situation have contributed to achieving "
        "victory.\n"
        "The basic information for the evaluation is as follows.\n"
        f"- Scenario : {scenario}\n"
        f"- Allied Team Agent Configuration : {ally_config}.\n"
        f"- Enemy Team Agent Configuration : {enemy_config}.\n"
        "- Situation Description : The situation involves the allied team and the "
        "enemy team engaging in combat, where victory is achieved by defeating all "
        "the enemies.\n"
        f"- Objective : {objective}\n"
        "* Important Notice : You should prefer the trajectory where our allies' "
        "health is preserved while significantly reducing the enemy's health. In "
        "similar situations, you should prefer shorter trajectory lengths.\n"
        "I will provide you with two trajectories, and you should select the better "
        "trajectory based on the outcomes of these trajectories. Regarding the "
        "trajectory, it will inform you about the final states, and you should "
        "select the better case based on these two trajectories.\n"
        f"{fmt(traj1, 1)}{fmt(traj2, 2)}"
        "Your task is to inform which one is better between [Trajectory1] and "
        "[Trajectory2] based on the information mentioned above. For example, if "
        "[Trajectory 1] seems better, output #1, and if [Trajectory 2] seems "
        "better, output #2. If it's difficult to judge or they seem similar, "
        "please output #0.\n"
        "* Important : Generally, it is considered better when fewer allied agents "
        "are killed or injured while inflicting more damage on the enemy.\n"
        "Omit detailed explanations and just provide the answer.")


def parse_llm_preference(response: str) -> float:
    """Map an LLM '#1'/'#2'/'#0' answer to a label (P(sigma1 preferred))."""
    r = response.strip()
    if "#1" in r:
        return 1.0
    if "#2" in r:
        return 0.0
    return 0.5  # '#0' tie / unsure


def annotate_with_llm(prompts: Sequence[str], model: str = "gpt-4o",
                      client=None) -> List[float]:
    """Annotate prompts with an OpenAI chat model (optional dependency).

    For large jobs use the OpenAI Batch API (the paper reports ~$42 for all
    SMAC tasks). This helper runs prompts sequentially as a reference.
    """
    if client is None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("pip install openai to use annotate_with_llm.") from e
        client = OpenAI()
    labels = []
    for p in prompts:
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": p}],
            temperature=0.0)
        labels.append(parse_llm_preference(resp.choices[0].message.content))
    return labels
