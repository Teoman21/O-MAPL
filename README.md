# O-MAPL: Offline Multi-Agent Preference Learning

A faithful, self-contained PyTorch implementation of **"O-MAPL: Offline
Multi-agent Preference Learning"** (Bui, Mai & Nguyen, 2025 — [arXiv:2501.18944](https://arxiv.org/abs/2501.18944)),
intended as a research baseline.

O-MAPL is an **end-to-end, single-phase** preference-based MARL method: instead
of first fitting a reward model and then running MARL (the two-phase approach),
it directly learns a **soft Q-function from pairwise trajectory preferences** by
exploiting the reward↔Q relationship in MaxEnt RL, then extracts decentralised
policies. It operates under **CTDE** with a carefully designed **linear value
factorisation** that preserves convexity of the learning objective and
global-local consistency (GLC).

## Install & run (no StarCraft/MuJoCo needed)

```bash
pip install -r requirements.txt          # torch, numpy, pyyaml

# 1) Generate a small synthetic preference dataset (already shipped as
#    data/synthetic.pkl; regenerate with:)
python scripts/make_synthetic_data.py --out data/synthetic.pkl

# 2) Train O-MAPL end-to-end and evaluate on the synthetic cooperative env
python -m omapl.train --config configs/synthetic.yaml

# 3) Baselines (same data/config, just change the algo)
python -m omapl.train --config configs/synthetic.yaml algo=ipl_vdn
python -m omapl.train --config configs/synthetic.yaml algo=iipl
python -m omapl.train --config configs/synthetic.yaml algo=bc

# Tests (component-level correctness + an end-to-end learning check)
python tests/test_components.py
python tests/test_smoke.py
```

On the synthetic `CoordinationEnv` (known optimum), O-MAPL reaches the optimal
return / 100% win rate, confirming the full pipeline learns.

## The algorithm (Section 5 / Algorithm 1)

Local networks (decentralisable): `q_i(o_i,a_i|ψ_q)`, `v_i(o_i|ψ_v)`,
`π_i(a_i|o_i;ω_i)`. Global values via a **single-layer (linear) mixing network**
with **non-negative** hypernetwork weights (Eqs. 6–7):

```
Q_tot(o,a) = Σ_i w^q_i(o,a) · q_i(o_i,a_i) + b^q       V_tot(o) = Σ_i w^v_i(o) · v_i(o_i) + b^v
R(o,a,o')  = Q_tot(o,a) − γ · V_tot(o')                 (inverse soft Bellman)
```

Each training step performs the **three alternating updates** of Algorithm 1:

| Step                           | Loss                                                                 | Updates          | Code                         |
| ------------------------------ | -------------------------------------------------------------------- | ---------------- | ---------------------------- |
| (4) Preference likelihood`L` | Bradley-Terry over`Σ_σ R` + χ² reg. `φ(x)=−½x²+x`        | `ψ_q`, `θ` | `OMAPL._update_preference` |
| (5) Extreme-V`J`             | `E[exp((Q_tot−V_tot)/β) − (Q_tot−V_tot)/β] − 1` (XQL/Gumbel) | `ψ_v`         | `OMAPL._update_extreme_v`  |
| (6) Local weighted-BC`Ψ`    | `E_{o,a}[ exp((Q_tot−V_tot)/β) · log π_i(a_i                     | o_i) ]`          | `ω`                       |

**Why these design choices (theory):**

- *Linear* mixing makes `L` concave in `q,θ` and `J` convex in `v`
  (Prop. 4.1); a 2-layer mixer breaks this (Prop. 4.2). → `Mixer.combine` is
  linear in the local values; verified in `tests/test_components.py`.
- *Non-negative* weights + linearity give global-local consistency
  (Thms. 4.3/4.4): the product of local WBC optima is the global WBC optimum.
- Minimising `J` makes `V_tot` the soft value `β log Σ_a μ(a|s)e^{Q_tot/β}`
  (verified: the Extreme-V test recovers the log-sum-exp).

**Gradient routing** (matching Algorithm 1, enforced and unit-tested):
`L` updates only `(ψ_q, θ)` — the bootstrap `V_tot(o')` uses **target** nets
(soft-updated with `tau`), so `θ` flows through the Q path; `J` updates only
`ψ_v` (Q detached, V-mixer weights detached via `Mixer.v_value_only`); `Ψ`
updates only `ω` (advantage weight detached).

Hyperparameters follow **Table 6** (`lr=1e-4`, `tau=0.005`, `γ=0.99`,
`batch=32`, agent hidden `256`, mixer hidden `64`). The MaxEnt temperature `β`
is **not specified in the paper** — it defaults to `1.0` and should be tuned.

## Baselines (Appendix B.5)

| Algo        | Description                                                        | Code                         |
| ----------- | ------------------------------------------------------------------ | ---------------------------- |
| `omapl`   | O-MAPL (linear hypernetwork mixer)                                 | `algos/omapl.py::OMAPL`    |
| `ipl_vdn` | O-MAPL with a VDN sum mixer (no hypernetwork)                      | `algos/omapl.py::IPL_VDN`  |
| `iipl`    | Independent IPL — single-agent IPL per agent, no mixing           | `algos/baselines.py::IIPL` |
| `bc`      | Behaviour cloning of preferred trajectories                        | `algos/baselines.py::BC`   |
| `sl_marl` | Two-phase reward-learning + OMIGA —*not included* (needs OMIGA) | —                           |

## Repository layout

```
omapl/
  networks/   agent.py (local q_i, v_i), policy.py (Categorical/Gaussian), mixer.py (linear hypernet + VDN)
  algos/      omapl.py (O-MAPL, IPL-VDN), baselines.py (IIPL, BC)
  data/       preference_dataset.py, generate_preferences.py (rule-based + LLM Table-5 prompt), omiga_adapter.py
  envs/       base.py, synthetic.py (CoordinationEnv), smac_wrapper.py, mamujoco_wrapper.py
  train.py    evaluate.py  utils/(config, logger, torch_utils)
configs/      synthetic.yaml, smacv2_protoss_5_vs_5.yaml
scripts/      make_synthetic_data.py
tests/        test_components.py, test_smoke.py
```

## Reproducing the paper's benchmarks (SMACv2 / SMACv1 / MAMuJoCo)

The algorithm is benchmark-agnostic. To run the paper's experiments you supply
(a) an offline preference dataset and (b) an evaluation environment:

1. **Offline data.** Obtain the offline buffers used in the paper — OMIGA
   (SMACv1, MAMuJoCo) and ComaDICE (SMACv2) — and convert them with
   `omapl.data.omiga_adapter.trajectories_from_arrays` (+ `load_hdf5`).
2. **Preference labels.** Use `omapl.data.generate_preferences`:
   *rule-based* (`build_preference_dataset` over poor/medium/expert), or
   *LLM-based* (`build_smac_preference_prompt` reproduces the **exact Table 5
   prompt**; `annotate_with_llm` queries GPT-4o). Save a `PreferenceDataset`.
3. **Eval env.** Install `smacv2` + StarCraft II (`envs/smac_wrapper.py`) or
   `multiagent_mujoco` (`envs/mamujoco_wrapper.py`). These are optional and
   import lazily; training does not require them.
4. Point a config's `data_path`/`task`/`env` at your dataset and run
   `python -m omapl.train --config configs/smacv2_protoss_5_vs_5.yaml`.

The dataset dimensions (`n_agents`, `obs_dim`, `state_dim`, `action_dim`) are
auto-filled from the `PreferenceDataset`.

## Implementation notes / fidelity

- **Mixer conditioning.** The Q-mixer hypernetwork conditions on the global
  state *and* joint action (`W^{o,a}_θ`, Eq. 7); the V-mixer on the global
  state (`W^o_θ`, Eq. 6). Toggle with `mixer_q_use_action`. The two heads live
  in one module (shared `θ`).
- **Discrete policies** mask unavailable actions to probability zero before the
  softmax (App. B.3); **continuous policies** use a diagonal Gaussian.
- **Parameter sharing** across agents with a one-hot agent id is on by default
  (`param_sharing`, `use_agent_id`), the standard SMAC setup; set
  `param_sharing=false` for independent per-agent networks.
- `β` (temperature) and `reg_coef` (χ² weight) are exposed; the paper does not
  report `β`, so treat it as a tuning knob per task.

## Citation

```bibtex
@article{bui2025omapl,
  title  = {O-MAPL: Offline Multi-agent Preference Learning},
  author = {Bui, The Viet and Mai, Tien and Nguyen, Thanh Hong},
  journal= {arXiv preprint arXiv:2501.18944},
  year   = {2025}
}
```
