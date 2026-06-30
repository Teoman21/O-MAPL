"""Build an O-MAPL preference dataset from a public OG-MARL SMACv2 Vault.

Pipeline (run on the HPC node where the Vault was downloaded):

    1. Download the Vault (see scripts/download_smacv2_data.sh), e.g.
         ./vaults/og_marl/smac_v2/terran_5_vs_5.vlt   (uids Good/Medium/Poor)
    2. Convert + label:
         python scripts/make_smacv2_data.py \
             --scenario terran_5_vs_5 \
             --vault_base ./vaults \
             --out data/smacv2_terran_5_vs_5.pkl
    3. Train:
         python -m omapl.train --config configs/smacv2_terran_5_vs_5.yaml

Labelling is rule-based across the OG-MARL quality tiers (Poor < Medium < Good),
matching the paper's poor/medium/expert scheme: cross-tier pairs are labelled by
quality, same-tier pairs by episodic return. 2000 pairs per task (Table 3).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omapl.data.generate_preferences import build_preference_dataset
from omapl.data.ogmarl_adapter import (
    OGMARL_QUALITY_ORDER, OGMARL_SMACV2_SCENARIOS, load_trajs_by_quality)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, choices=OGMARL_SMACV2_SCENARIOS,
                   help="OG-MARL SMACv2 scenario (the only three available).")
    p.add_argument("--vault_base", default="./vaults",
                   help="Base dir passed to download_vault.py.")
    p.add_argument("--out", default=None,
                   help="Output .pkl (default data/smacv2_<scenario>.pkl).")
    p.add_argument("--n_per_tier", type=int, default=1000,
                   help="Trajectories kept per quality tier (~paper's 1k).")
    p.add_argument("--n_pairs", type=int, default=2000,
                   help="Preference pairs to generate (Table 3).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = args.out or f"data/smacv2_{args.scenario}.pkl"
    vault_rel_dir = os.path.join(args.vault_base, "og_marl", "smac_v2")
    vault_name = f"{args.scenario}.vlt"
    rng = np.random.default_rng(args.seed)

    trajs_by_quality, spec = load_trajs_by_quality(
        vault_rel_dir, vault_name, OGMARL_QUALITY_ORDER,
        n_per_tier=args.n_per_tier, rng=rng)
    present = [t for t in OGMARL_QUALITY_ORDER if t in trajs_by_quality]
    print(f"spec: n_agents={spec.n_agents} obs_dim={spec.obs_dim} "
          f"state_dim={spec.state_dim} action_dim={spec.action_dim}")
    for t in present:
        rets = [tr.get("return", 0.0) for tr in trajs_by_quality[t]]
        print(f"  {t:7s}: mean return {np.mean(rets):.2f} over {len(rets)} trajs")

    ds = build_preference_dataset(
        trajs_by_quality, present, args.n_pairs, spec, rng=rng)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    ds.save(out)
    print(f"Saved {len(ds)} preference pairs -> {out}")


if __name__ == "__main__":
    main()
