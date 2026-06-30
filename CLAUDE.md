# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch, faithful implementation of the paper **O-MAPL: Offline
Multi-agent Preference Learning** (arXiv:2501.18944, `paper.pdf` in the repo
root). It is meant to be an accurate research **baseline**. When changing the
algorithm, cross-check against the paper — the mapping from equations to code is
documented in `README.md` and in module docstrings (which cite specific
equations/propositions). Do not "improve" the math away from the paper without
flagging it; fidelity is the goal.

## Commands

```bash
pip install -r requirements.txt                         # torch, numpy, pyyaml
python scripts/make_synthetic_data.py --out data/synthetic.pkl   # build sample data
python -m omapl.train --config configs/synthetic.yaml   # train (algo=omapl|ipl_vdn|iipl|bc)
python -m omapl.train --config configs/synthetic.yaml algo=ipl_vdn seed=1   # CLI overrides: key=value
python tests/test_components.py                         # unit tests (no pytest needed)
python tests/test_smoke.py                              # end-to-end learning check
```

Configs are YAML (`configs/`); any field is overridable on the CLI as
`key=value` positional args. Use `python3` on this machine (`python` is absent).

## Architecture (big picture)

End-to-end preference→policy learning, no explicit reward model. One training
step (`*.update(batch)`) runs **three alternating updates** (Algorithm 1):

1. **Preference loss** `L` (Bradley-Terry over per-trajectory sums of the
   inverse-soft-Bellman reward `R = Q_tot − γ·V_tot(o')`, + χ² regulariser) →
   updates local Q (`ψ_q`) and mixer (`θ`).
2. **Extreme-V loss** `J` (XQL/Gumbel) → updates local V (`ψ_v`), making
   `V_tot` the soft (log-sum-exp) value.
3. **Local weighted-BC** `Ψ` (advantage-weighted log-likelihood) → updates the
   decentralised policies (`ω`).

The non-obvious, must-preserve invariants:
- **Mixing is linear** in the local values with **non-negative** hypernetwork
  weights (`networks/mixer.py`). Linearity ⇒ convex objective (Prop. 4.1);
  non-negativity ⇒ global-local consistency (Thms. 4.3/4.4). A 2-layer mixer
  would break both — keep it single-layer.
- **Gradient routing** is strict (and unit-tested in
  `test_components.py::test_gradient_routing`): `L` must not move `ψ_v` or the
  policy; `J` must not move `ψ_q`/`θ`; `Ψ` moves only the policy. This is
  achieved with target nets for the `V_tot(o')` bootstrap and explicit
  `detach()` / `Mixer.v_value_only`. If you refactor the learner, re-run that
  test.

Data flows as `PreferenceDataset` (trajectories stored once, pairs reference
them by index; batches are padded to `[B, 2, T, ...]` with a `mask`). All
learners consume the same batch dict. Envs (`envs/`) are **only** for
evaluation and import lazily — SMAC/MAMuJoCo are optional; the synthetic
`CoordinationEnv` always works.

## Key fidelity knobs

- `beta` (MaxEnt temperature) is **not given in the paper** — default 1.0, tune
  per task. `reg_coef` is the χ² weight.
- Discrete policies mask unavailable actions (SMAC); continuous use a Gaussian.
- `param_sharing` + `use_agent_id` (default on) is the standard SMAC setup.
- Real benchmarks need external data: convert OMIGA/ComaDICE buffers via
  `data/omiga_adapter.py`, then label with `data/generate_preferences.py`
  (rule-based, or the exact Table-5 GPT-4o prompt).
