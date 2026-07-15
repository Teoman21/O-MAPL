# O-MAPL implementation — status notes

A short, honest account of what this codebase is, what has been verified, and
what remains. Intended as a handoff alongside the code.

## What it is

A faithful from-scratch reimplementation of **O-MAPL** (Bui, Mai & Nguyen,
2025; arXiv:2501.18944). There is no public reference code, so it is built
directly from the paper's equations. It is intended as a **baseline**.

The full method is implemented:

- The **linear value-factorization mixer** with non-negative hypernetwork
  weights (Eqs. 6–7).
- The **three alternating updates** of Algorithm 1:
  1. Bradley-Terry **preference loss** with the χ² regularizer `φ(x)=−½x²+x`,
  2. the **Extreme-V / Gumbel loss** (XQL),
  3. **local weighted behavior cloning** for decentralized policy extraction.
- Baselines: **IPL-VDN**, **IIPL**, **BC** (Appendix B.5).

## What has been verified (and how strong each check is)

**Level 1 — correctness against the paper's theory.** These checks have an
*external* ground truth, so they genuinely catch bugs
(`tests/test_components.py`):

- **Extreme-V → soft value.** Optimizing against the `J` loss converges to the
  log-sum-exp soft value, compared against an independent computation
  (`v = 0.991` vs `target = 0.991`). Confirms `J` does what the paper claims.
- **Mixer linearity + non-negativity.** Verifies the precondition for the
  convexity result (Prop. 4.1) and global-local consistency (Thms. 4.3/4.4).
- **Gradient routing matches Algorithm 1.** The preference loss updates only Q
  and the mixer; Extreme-V only V; weighted-BC only the policy. This
  detach/target-network structure is the easy thing to get wrong, and it is
  unit-tested.

**Level 2 — end-to-end learning** (`tests/test_smoke.py`). On a synthetic
cooperative task with a *known* optimum, O-MAPL recovers the optimal
decentralized policy from preference labels alone: return goes from ~2.8
(random) to the optimum of 10. The baselines also reach the optimum.

So **"verified" means**: the algorithm is faithful to the equations, the
theoretical properties the paper proves actually hold in the code, and the full
pipeline learns. It does **not** mean the benchmark numbers are reproduced.

## What has NOT been verified

- **The paper's benchmark results (Tables 1–2) on SMAC/MAMuJoCo are not
  reproduced.** That requires the real OMIGA / ComaDICE offline datasets and
  StarCraft II / MuJoCo, which are not wired in yet. Adapters and env wrappers
  are in place (`omapl/data/omiga_adapter.py`, `omapl/envs/`), but no real run
  has been done.

## Interpretation points (paper underspecified)

Three judgment calls that could differ from the authors' intent; reconcile
these first if reference code is released:

1. **Temperature `β`** — not reported in the paper. Defaults to `1.0`; treat as
   a per-task tuning knob.
2. **Q-mixer action conditioning** — the Q-mixer hypernetwork is conditioned on
   the global state *and* joint action (`W^{o,a}_θ`, Eq. 7). Toggle:
   `mixer_q_use_action`.
3. **χ² regularizer scope** — applied over all transitions of both trajectories
   in each pair (the practical `Σ_P φ(R)` form).

## Next step to close the loop to the paper

Convert one SMACv2 offline dataset (ComaDICE) via `omiga_adapter`, label it with
`generate_preferences` (rule-based or the Table-5 GPT-4o prompt), and run it
against the real SMACv2 env. Success there = win rate matches Table 1 and beats
IPL-VDN / IIPL / BC.

## How to re-run the verification

```bash
python3 tests/test_components.py   # Level 1: theory/correctness
python3 tests/test_smoke.py        # Level 2: end-to-end learning
python3 -m omapl.train --config configs/synthetic.yaml   # watch training metrics
```
