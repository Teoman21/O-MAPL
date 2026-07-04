"""Adapter for the public OG-MARL (Off-the-Grid MARL) SMACv2 offline datasets.

The O-MAPL paper sources its SMACv2 buffers from ComaDICE, which were never
publicly released. The closest *public* SMACv2 offline data is OG-MARL
(InstaDeep), hosted on Hugging Face as flashbax ``Vault``s. It covers three of
the paper's fifteen cells:

    smac_v2/terran_5_vs_5, smac_v2/zerg_5_vs_5, smac_v2/terran_10_vs_10

each with quality levels ``Good`` / ``Medium`` / ``Poor`` (and ``Replay``).
This module loads such a Vault and converts it into the trajectory + DataSpec
form consumed by :func:`omapl.data.generate_preferences.build_preference_dataset`.

OG-MARL Vault layout (verified against the OG-MARL dataset API demo): the
experience is a single concatenated sequence with a leading batch dim ``B=1``::

    observations        (1, T, N, obs_dim)
    actions             (1, T, N)              discrete action indices
    rewards             (1, T, N)              shared reward, replicated per agent
    terminals           (1, T, N)              1.0 at a true episode terminal
    truncations         (1, T, N)              1.0 at a timeout boundary
    infos.legals        (1, T, N, n_actions)   available-action masks
    infos.state         (1, T, state_dim)      global state

Episodes are delimited by ``terminals | truncations``. We treat an episode's
last step as ``done`` (the O-MAPL learner masks the ``gamma * V_tot(o')``
bootstrap by ``1 - done`` — see ``omapl/algos/omapl.py``); the bootstrapped
``next_*`` at that final step is therefore irrelevant and we simply duplicate
the last frame.

Note: OG-MARL data is *not* the same as the ComaDICE buffers used in the paper,
so absolute win rates will differ from Table 1 / the paper figure. This path
reproduces the *method and curves* on real public SMACv2 data for the three
available cells.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .preference_dataset import DataSpec
from .omiga_adapter import trajectories_from_arrays


# Quality levels, worst -> best. Maps onto the paper's poor/medium/expert tiers.
OGMARL_QUALITY_ORDER: Tuple[str, ...] = ("Poor", "Medium", "Good")

# The three SMACv2 scenarios OG-MARL actually provides.
OGMARL_SMACV2_SCENARIOS: Tuple[str, ...] = (
    "terran_5_vs_5", "zerg_5_vs_5", "terran_10_vs_10",
)


def load_ogmarl_vault(vault_rel_dir: str, vault_name: str, vault_uid: str) -> Dict:
    """Read one OG-MARL Vault quality split into a numpy experience dict.

    ``vault_rel_dir`` / ``vault_name`` / ``vault_uid`` map directly onto
    ``flashbax.vault.Vault(rel_dir=..., vault_name=..., vault_uid=...)``. After
    ``download_vault.py`` the on-disk path is
    ``<base>/og_marl/smac_v2/<scenario>.vlt`` with uids ``Good/Medium/Poor``.
    """
    try:
        from flashbax.vault import Vault  # type: ignore
        import jax  # type: ignore
    except ImportError as e:  # pragma: no cover - optional, HPC-only dependency
        raise ImportError(
            "OG-MARL loading needs flashbax + jax: pip install flashbax jax jaxlib"
        ) from e
    vault = Vault(rel_dir=vault_rel_dir, vault_name=vault_name, vault_uid=vault_uid)
    exp = vault.read().experience
    return jax.tree_util.tree_map(lambda x: np.asarray(x), exp)


def experience_to_arrays(exp: Dict) -> Dict[str, np.ndarray]:
    """Convert an OG-MARL experience dict to the flat-array form the OMIGA
    adapter expects. Strips the leading ``B=1`` dim and builds ``next_*`` by a
    one-step shift (the final frame is duplicated; it is masked by ``done``)."""

    def first(x):
        return np.asarray(x)[0]  # drop leading batch dim

    obs = first(exp["observations"]).astype(np.float32)          # [T, N, obs]
    actions = first(exp["actions"])                              # [T, N]
    rewards = first(exp["rewards"]).astype(np.float32)           # [T, N]
    terminals = first(exp["terminals"]).astype(np.float32)       # [T, N]
    infos = exp["infos"]
    legals = first(infos["legals"]).astype(np.float32)           # [T, N, A]
    state = first(infos["state"]).astype(np.float32)             # [T, state]

    if "truncations" in exp:
        truncations = first(exp["truncations"]).astype(np.float32)
    else:
        truncations = np.zeros_like(terminals)

    # Episode boundary = a true terminal OR a timeout truncation (any agent).
    ep_end = ((terminals.max(axis=-1) > 0.5) |
              (truncations.max(axis=-1) > 0.5)).astype(np.float32)  # [T]
    # Reward is shared across agents in SMAC; take a single column.
    reward_t = rewards[:, 0]                                      # [T]

    def shift(x):
        return np.concatenate([x[1:], x[-1:]], axis=0)

    return {
        "obs": obs,
        "actions": actions.astype(np.float32),
        "next_obs": shift(obs),
        "state": state,
        "next_state": shift(state),
        "terminals": ep_end,
        "avail": legals,
        "next_avail": shift(legals),
        "rewards": reward_t,
    }


def trajectories_from_vault(vault_rel_dir: str, vault_name: str, vault_uid: str
                            ) -> Tuple[List[Dict[str, np.ndarray]], DataSpec]:
    """Load one quality split and slice it into per-episode trajectory dicts."""
    exp = load_ogmarl_vault(vault_rel_dir, vault_name, vault_uid)
    arr = experience_to_arrays(exp)
    return trajectories_from_arrays(
        obs=arr["obs"], actions=arr["actions"], next_obs=arr["next_obs"],
        state=arr["state"], next_state=arr["next_state"],
        terminals=arr["terminals"], discrete=True,
        avail=arr["avail"], next_avail=arr["next_avail"], rewards=arr["rewards"],
    )


def list_vault_uids(vault_rel_dir: str, vault_name: str) -> List[str]:
    """Return the uid subdirectories present in a vault dir on disk.

    A flashbax vault is laid out ``<rel_dir>/<vault_name>/<uid>/``. OG-MARL's
    original datasets used quality uids (``Good``/``Medium``/``Poor``); the
    public ``core/smac_v2`` vaults ship a single combined uid (``Replay``).
    """
    import os
    vlt = os.path.join(vault_rel_dir, vault_name)
    if not os.path.isdir(vlt):
        return []
    return sorted(d for d in os.listdir(vlt)
                  if os.path.isdir(os.path.join(vlt, d)))


def load_single_buffer_as_tiers(
    vault_rel_dir: str, vault_name: str, vault_uid: str,
    tier_names: Sequence[str] = ("poor", "medium", "expert"),
    n_per_tier: Optional[int] = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Dict[str, List[Dict]], DataSpec]:
    """Reconstruct poor/medium/expert tiers from a *single* combined buffer.

    The public OG-MARL ``core/smac_v2`` vaults ship one mixed-quality buffer
    (uid ``Replay``), not the Good/Medium/Poor splits our other loader expects,
    and not the (unreleased) ComaDICE quality tiers the paper used. To keep the
    paper's rule-based preference scheme applicable, we load the single buffer,
    slice it into per-episode trajectories, and bucket episodes into quality
    tiers by **episodic-return terciles** (lowest third -> worst tier). This is a
    documented, transparent deviation forced by the public data; the labelling
    itself (cross-tier by tier, same-tier by return) is unchanged from the paper.

    Returns ``({tier: [traj, ...]}, spec)`` with tiers ordered worst -> best.
    """
    rng = rng or np.random.default_rng()
    trajs, spec = trajectories_from_vault(vault_rel_dir, vault_name, vault_uid)
    # Rank episodes by return (ascending) and split into equal terciles.
    order = sorted(range(len(trajs)), key=lambda i: trajs[i].get("return", 0.0))
    n = len(order)
    cut1, cut2 = n // 3, (2 * n) // 3
    buckets: Dict[str, List[Dict]] = {t: [] for t in tier_names}
    for rank, i in enumerate(order):
        tier = (tier_names[0] if rank < cut1
                else tier_names[1] if rank < cut2 else tier_names[2])
        buckets[tier].append(trajs[i])
    if n_per_tier is not None:
        for t in tier_names:
            if len(buckets[t]) > n_per_tier:
                keep = rng.choice(len(buckets[t]), size=n_per_tier, replace=False)
                buckets[t] = [buckets[t][j] for j in keep]
    return buckets, spec


def load_trajs_by_quality(
    vault_rel_dir: str, vault_name: str,
    quality_order: Sequence[str] = OGMARL_QUALITY_ORDER,
    n_per_tier: Optional[int] = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Dict[str, List[Dict]], DataSpec]:
    """Load every available quality split and (optionally) subsample episodes.

    Returns ``({tier: [traj, ...]}, spec)``. The paper uses ~1k trajectories per
    quality level, so ``n_per_tier`` defaults to 1000. Missing tiers are skipped
    with a warning rather than failing.
    """
    rng = rng or np.random.default_rng()
    trajs_by_quality: Dict[str, List[Dict]] = {}
    spec: Optional[DataSpec] = None
    for tier in quality_order:
        try:
            trajs, sp = trajectories_from_vault(vault_rel_dir, vault_name, tier)
        except FileNotFoundError:
            print(f"[ogmarl] quality '{tier}' not found for {vault_name}; skipping.")
            continue
        spec = spec or sp
        if n_per_tier is not None and len(trajs) > n_per_tier:
            keep = rng.choice(len(trajs), size=n_per_tier, replace=False)
            trajs = [trajs[i] for i in keep]
        trajs_by_quality[tier] = trajs
        print(f"[ogmarl] {vault_name} [{tier}]: {len(trajs)} trajectories")
    if spec is None:
        raise RuntimeError(f"No quality splits loaded for {vault_name}.")
    return trajs_by_quality, spec
