# Paper → Code Map (O-MAPL, arXiv:2501.18944)

A line-by-line bridge from the paper (`paper.pdf`) to this re-implementation. The
authors released no reference code, so this file is the record that lets us say
**exactly** which equation / proposition / table each piece of code implements,
and which choices are *ours* (paper underspecified).

How to read each row: **Paper object** → *what it says* → **`file:line`** →
*how the code does it / any deviation*. Citations are to the arXiv v1 PDF.

---

## 0. The one-paragraph summary

O-MAPL never learns a reward model. It uses the MaxEnt RL identity
`r = Q − γ·E[V(s′)]` (inverse soft Bellman) to turn *reward learning from
preferences* into *soft-Q learning*. A **linear, non-negative mixing network**
factorises global `Q_tot`/`V_tot` into per-agent `q_i`/`v_i` (so execution is
decentralised and the objective stays convex). Each training step runs **three
alternating updates** (Algorithm 1): a Bradley-Terry **preference loss** (learns
`q`, mixer `θ`), an **Extreme-V/Gumbel loss** (makes `V_tot` the soft value),
and a **weighted behaviour-cloning loss** (extracts decentralised policies).

---

## 1. Section 3 — Background (MaxEnt RL machinery)

| Paper object | Statement | Code | Notes |
|---|---|---|---|
| Soft value `V_tot(s) = β log Σ_a μ(a\|s) e^{Q_tot/β}` (p.3) | `V_tot` is the log-sum-exp of `Q_tot` | enforced *indirectly* by the Extreme-V loss → `omapl/algos/omapl.py:140` (`_update_extreme_v`) | We don't compute the log-sum-exp directly (intractable for continuous actions); XQL does it for us. Verified in `tests/test_components.py:68`. |
| Eq (1) soft policy `π*_tot = μ·exp((Q*−V*)/β)` (p.3) | optimal policy from `Q,V` | weight `exp((Q_tot−V_tot)/β)` in `omapl/algos/omapl.py:164` | We never form `π*` explicitly; the exponential advantage becomes the WBC weight (Eq 3/4). |
| Inverse soft Bellman `(T*Q)(s,a) = Q − γ·E_{s′}V(s′)` (p.3) | one-to-one map `r ↔ Q` | `R = q_tot − γ·v_next·(1−done)` → `omapl/algos/omapl.py:113`; diagnostic copy `recovered_reward()` `:187` | This `R` *is* the recovered reward (Table 7). `(1−done)` zeros the bootstrap past terminals — a standard, paper-implicit detail. |
| KL-to-behaviour MaxEnt objective (p.3) | `β` regularises toward `μ_tot` | temperature `cfg.beta`, `omapl/utils/config.py:57` | **β is NOT in the paper.** Default `1.0`, tune per task. This is interpretation knob #1. |

---

## 2. Section 4.1 — Preference-based inverse Q-learning

| Paper object | Statement | Code | Notes |
|---|---|---|---|
| Bradley-Terry over trajectory reward sums (p.3) | `P(σ1≻σ2) = softmax(Σ_σ r)` | `S = (R*mask).sum(dim=2); logp = log_softmax(S)` → `omapl/algos/omapl.py:116-117` | Sum of per-step `R` over each trajectory, then a 2-way softmax. |
| `P(σ1≻σ2 \| Q_tot)` in Q-space (p.4) | replace `r` by `(T*Q_tot)` | same as above — `S` is built from `R=(T*Q_tot)` | This is the key "train in Q-space, not reward-space" step. |
| Soft cross-entropy label | preferred-trajectory likelihood | `bt_loss = −(label·logp[:,0] + (1−label)·logp[:,1]).mean()` → `omapl/algos/omapl.py:119` | `label∈{0, 0.5, 1}` (idx 0 = σ1). `0.5` handles LLM "#0" ties. |
| Data format `σ = {(s,a)…}`, dataset `P` (p.3) | pairs of trajectories | `PreferenceDataset` → `omapl/data/preference_dataset.py:47`; pairs are index refs, batches padded to `[B,2,T,…]` with `mask` | The `2` axis is the (σ1,σ2) pair; `mask` handles ragged lengths. |

---

## 3. Section 4.2 — Value factorisation & the two value losses

| Paper object | Statement | Code | Notes |
|---|---|---|---|
| `Q_tot = M_w[q]`, `V_tot = M_w[v]` | mixing network combines locals | `Mixer.mix_q` / `mix_v` → `omapl/networks/mixer.py:89,94`; called via `_q_tot`/`_v_tot` `omapl/algos/omapl.py:84-90` | — |
| `R_w[q,v] = M_w[q] − γ·E M_w[v(s′)]` (p.4) | mixed inverse soft Bellman | `omapl/algos/omapl.py:113` | bootstrap uses **target** nets (`target=True`) — see routing §6. |
| **Full objective `L(q,v,w)`** (p.4): `Σ_σ1 R − log(e^{Σσ1 R}+e^{Σσ2 R}) + φ(R)` | BT likelihood + regulariser | `_update_preference` → `omapl/algos/omapl.py:106-130` | maximised, so coded as `loss = bt_loss − reg_coef·reg` then minimised. |
| χ² regulariser `φ(x) = −½x² + x` (p.4) | concave, bounds rewards | `phi = −0.5*R.pow(2) + R` → `omapl/algos/omapl.py:122` | applied over **all** masked transitions of **both** trajectories (the practical `Σ_P φ(R)` form). Interpretation knob #3 (NOTES.md). |
| **Extreme-V loss `J(v)`** (p.4): `E[e^{(Q−V)/β}] − E[(Q−V)/β] − 1` | XQL/Gumbel, makes `V` the soft value | `gumbel = exp(z) − z − 1`, `z=(Q_tot−V_tot)/β` → `omapl/algos/omapl.py:148-150` | `exp_clip` (cfg `:62`) guards the exponential. Recovers `β·logsumexp(Q/β)` — proven by `tests/test_components.py:68` (`v=0.991` vs `target=0.991`). |
| `M_w[v] = β log Σ μ e^{M_w[q]/β}` (p.4) | desired fixed point of `J` | not coded directly; *is the minimiser* of `J` above | — |
| Alternating "update q,w" then "update v" (p.4) | training schedule | `update()` calls `_update_preference` then `_update_extreme_v` → `omapl/algos/omapl.py:96-103` | matches Algorithm 1 lines 4–5. |
| **Prop 4.1 (convexity)** | `L` concave in `q,w`; `J` convex in `v` — *only if mixing linear* | `Mixer.combine` is linear: `(w*local).sum() + b` → `omapl/networks/mixer.py:83-87` | Linearity is the load-bearing property. Tested: `tests/test_components.py:55` (`mix(αq1+(1−α)q2)=α mix(q1)+(1−α)mix(q2)`). |
| **Prop 4.2 (non-convexity, 2-layer)** | a 2-layer mixer breaks Prop 4.1 | enforced by *design*: single-layer mixer; hypernets may be deep but `combine` stays linear in `q,v` (docstring `omapl/networks/mixer.py:1-21`) | "Do not make the mixer 2-layer" — CLAUDE.md invariant. |
| Non-negative weights | monotonic mixing (QMIX) | `.abs()` on hypernet output → `omapl/networks/mixer.py:72,78` | Tested: `tests/test_components.py:49` (`(w_q>=0).all()`). Underpins Thms 4.3/4.4. |

---

## 4. Section 4.3 — Local policy extraction (WBC)

| Paper object | Statement | Code | Notes |
|---|---|---|---|
| Eq (2) local-value extraction `π*_i ∝ μ_i e^{(w^q q_i − w^v v_i)/β}` | the *naïve* method the paper rejects | **not implemented** (intentionally) | Paper shows it can yield invalid (non-normalised) policies. We use WBC instead — matching the paper's chosen method. |
| Eq (3) global WBC: `max E[e^{(Q_tot−V_tot)/β} log π_tot]` | global advantage-weighted BC | conceptual target; realised per-agent below | — |
| **Eq (4) local WBC**: `max E[e^{(Q_tot−V_tot)/β} log π_i]` | per-agent loss with **global** weight | `Ψ` → `_update_policy` `omapl/algos/omapl.py:159-176` | weight `exp((Q_tot−V_tot)/β)` is **detached** (`no_grad`, `:161`); `log π_i` summed over agents `:170`. The weight uses *global* `Q_tot,V_tot` → credit assignment. |
| **Thm 4.3 (GLC)** | product of local WBC optima = global WBC optimum | guaranteed by using Eq (4) + non-negative linear mixer (above) | not a code line — a property we *rely on*; the non-negativity/linearity tests guard its preconditions. |
| **Thm 4.4 / Eq (5),(9)**: `π*_i = (η/Δ) μ_i e^{(w^q q_i − w^v v_i)/β}` | closed form with normalisation correction `η/Δ` | implicit: we *learn* `π_i` by gradient on Eq (4) instead of evaluating the closed form | The learned softmax policy *is* the normalised object; `η/Δ` is the normaliser we'd otherwise need (App A.4). |
| **Prop 4.5 / Eq (11)**: `v_i = (β/w^v_i) log Σ e^{w^q q_i/β} + …` | local `v_i` is a modified local log-sum-exp | not coded (theoretical justification only) | Explains why local `v_i` need not be the plain log-sum-exp of `q_i`. |

---

## 5. Section 5 — Practical Algorithm (this is what runs)

| Paper object | Statement | Code | Notes |
|---|---|---|---|
| `q_i(o_i,a_i\|ψ_q)`, `v_i(o_i\|ψ_v)` | local nets on **observations** (POMDP) | `LocalQNet` / `LocalVNet` → `omapl/networks/agent.py:61,101` | param-sharing + one-hot agent id (`:39,47`) — standard SMAC setup, knob `param_sharing`/`use_agent_id`. |
| **Eq (6)** `M_θ[v] = vᵀW^o_θ + b^o_θ` | V-mixer conditions on obs/state | `v_params` (hypernet on `state`) → `omapl/networks/mixer.py:76`; `hyper_w_v`,`hyper_b_v` `:52-53` | — |
| **Eq (7)** `M_θ[q] = qᵀW^{o,a}_θ + b^{o,a}_θ` | Q-mixer conditions on obs/state **and joint action** | `q_params` (hypernet on `[state, joint_action]`) → `omapl/networks/mixer.py:63-74`; action one-hot `_joint_action_feat` `:56` | Conditioning the Q-mixer on the action is interpretation knob #2 (`mixer_q_use_action`, cfg `:61`). Shared `θ` across both heads (one `Mixer` module). |
| **Algorithm 1** lines 4-6 | three alternating updates | `OMAPL.update()` → `omapl/algos/omapl.py:96-103` | line4→`_update_preference`, line5→`_update_extreme_v`, line6→`_update_policy`, then soft target update. |
| Practical `L(ψ_q,ψ_v,θ)` over `(o,a,o′)` | preference loss, observations | `_update_preference` `:106` | identical math to §3 row, on obs. |
| Practical `J(ψ_v)` | extreme-V on obs | `_update_extreme_v` `:140` | — |
| Practical `Ψ(ω_i)` | local WBC on obs | `_update_policy` `:159` | — |
| Algorithm 1 output `π_i(a_i\|o_i;ω_i)` | decentralised policies | `CategoricalPolicy`/`GaussianPolicy` → `omapl/networks/policy.py:25,58`; `act()` `omapl/algos/omapl.py:181` | discrete masks unavailable actions (App B.3); see §7. |

### 5.1 Gradient routing (Algorithm 1, enforced — the easy thing to get wrong)

| Update | Must move | Must NOT move | Mechanism | Code |
|---|---|---|---|---|
| `L` (pref) | `ψ_q`, `θ` (mixer) | `ψ_v`, policy | bootstrap `V_tot(o′)` via **target** v-net + target mixer (`no_grad`) | `omapl/algos/omapl.py:111-113`; opt only `q_net+mixer` `:68` |
| `J` (extreme-V) | `ψ_v` | `ψ_q`, `θ`, policy | `Q_tot` detached (`no_grad` `:142`); V-mixer weights frozen via `Mixer.v_value_only` (`.detach()`) `omapl/networks/mixer.py:98-106` | opt only `v_net` `:70` |
| `Ψ` (WBC) | `ω` (policy) | `ψ_q`,`ψ_v`,`θ` | advantage weight detached (`no_grad` `:161`) | opt only `policy` `:71` |
| target tracking | — | — | `soft_update(…, tau)` (Table 6 `tau=0.005`) | `omapl/algos/omapl.py:101-102` |

Unit-tested exactly: `tests/test_components.py:88` (`test_gradient_routing`) snapshots every param group and asserts which changed.

---

## 6. Section 6 + Appendix B — Experiments, data, baselines, hyperparameters

| Paper object | Code | Notes |
|---|---|---|
| **Baselines (App B.5)** | | |
|  IPL-VDN (VDN sum mixer, no hypernet) | `IPL_VDN(OMAPL)` → `omapl/algos/omapl.py:203`; `VDNMixer` `omapl/networks/mixer.py:109` | one-line subclass: `Q_tot=Σq_i`, `V_tot=Σv_i`. |
|  IIPL (independent single-agent IPL per agent, no mixing) | `IIPL` → `omapl/algos/baselines.py:30` | per-agent `R_i`, per-agent BT/Extreme-V/WBC; no mixer. |
|  BC (imitate preferred trajectory) | `BC` → `omapl/algos/baselines.py:119` | weight `[label, 1−label]`, no value learning. |
|  SL-MARL (two-phase reward+OMIGA) | **not included** | needs external OMIGA; out of scope (README §Baselines). |
| **Rule-based labels (App B.1)** poor/medium/expert, better tier preferred | `make_rule_based_pairs` → `omapl/data/generate_preferences.py:89` | cross-tier → label by tier; same-tier → by return / tie. |
| **LLM labels / Table 5 prompt (App B.2)** GPT-4o, final-state summary | `build_smac_preference_prompt` → `omapl/data/generate_preferences.py:136` (verbatim Table-5 template); `parse_llm_preference` `:200` (#1/#2/#0→1/0/0.5); `annotate_with_llm` `:210` | prompt text reproduces Table 5 exactly. |
| **Table 3 datasets** (dims, lengths) | `DataSpec` + auto-fill `fill_dims_from_dataset` → `omapl/train.py:40` | dims read from the dataset, not hard-coded. |
| **Table 6 hyperparameters** Adam, lr 1e-4, τ 0.005, γ 0.99, batch 32, agent hid 256, mixer hid 64, 32 eval eps, 100 eval steps | `Config` defaults → `omapl/utils/config.py:46-71` | match Table 6 1:1. `beta`/`reg_coef`/clips are *extra* knobs (not in Table 6). |
| **Table 7 recovered rewards** `R(o,a,o′)=M_θ[q]−γM_θ[v(o′)]` | `OMAPL.recovered_reward()` → `omapl/algos/omapl.py:187` | diagnostic: preferred trajs should get positive `R`, dispreferred negative. |
| **App B.3 discrete policy** softmax over *available* actions only | `CategoricalPolicy.logits` masks `avail<0.5` to `−1e10` → `omapl/networks/policy.py:33-38` | unavailable actions get prob 0. |
| **App B.3 continuous policy** `torch.distributions.Normal` | `GaussianPolicy` → `omapl/networks/policy.py:58` | state-dependent mean & log-std. |
| **App B.6 eval metrics** mean/std return, win-rate % | `evaluate_policy` → `omapl/evaluate.py:11`; win-rate ×100 `:38`; fully decentralised rollout `:26` | Tables 1-2 / 8-16 are the targets; **not reproduced** (need real OMIGA/ComaDICE buffers — see NOTES.md). |
| Algorithm 1 training loop ("certain number of steps") | `train()` loop → `omapl/train.py:90-106` | `n_train_steps`, periodic eval + best-checkpoint. |

---

## 7. Faithful vs. our interpretation (state this up front in any write-up)

**Faithful to the paper (and verified):**
- The three losses `L`, `J`, `Ψ` (Eqs in §4.2/4.3/5) and their gradient routing
  (Algorithm 1) — `tests/test_components.py::test_gradient_routing`.
- Linear, non-negative single-layer mixer (Eqs 6-7; Props 4.1/4.2; Thms 4.3/4.4)
  — `test_mixer_linear_in_inputs`, `test_mixer_shapes_and_nonneg`.
- Extreme-V → soft (log-sum-exp) value — `test_extreme_v_recovers_logsumexp`.
- Table-5 LLM prompt verbatim; Table-6 hyperparameters 1:1.

**Our choices where the paper is silent (reconcile first if reference code appears):**
1. **`β` temperature** — not reported. Default `1.0`; per-task knob. (`config.py:57`)
2. **Q-mixer action conditioning** — we feed the joint action to the Q-mixer
   hypernet (Eq 7 reads `W^{o,a}_θ`). Toggle `mixer_q_use_action`. (`mixer.py:63`)
3. **χ² regulariser scope** — applied over all transitions of both trajectories
   in a pair (the practical `Σ_P φ(R)` form). (`omapl.py:122`)
4. Stability clips (`exp_clip`, `adv_weight_clip`, `grad_clip`) — engineering,
   not in the paper. (`config.py:62-64`)

**Not reproduced:** the benchmark numbers (Tables 1-2, 8-16). Algorithm fidelity
and the theoretical properties are verified; matching SMAC/MAMuJoCo win-rates
needs the unreleased OMIGA/ComaDICE offline buffers (see `REPRODUCE_SMACV2.md`).
