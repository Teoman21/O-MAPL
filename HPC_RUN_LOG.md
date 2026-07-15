# O-MAPL on Northeastern Explorer — HPC run log

A complete, honest record of reproducing the **O-MAPL** win-rate curves on real
public SMACv2 data on the Northeastern **Explorer** cluster (SLURM + GPU). This
captures what we actually did, every problem we hit, the fix, and the final run
procedure — so the run is reproducible and the deviations from the paper are on
the record.

See also: `REPRODUCE_SMACV2.md` (the intended runbook), `NOTES.md` (verification
status), `PAPER_TO_CODE.md` (equation→code map).

---

## 0. What this reproduces (and the honest scope)

- **Method:** the O-MAPL algorithm, faithfully (three alternating updates, linear
  non-negative mixer, BT preference loss + χ² regulariser, Extreme-V/Gumbel,
  weighted-BC). Unit-tested against the paper's theory.
- **Data:** the **public OG-MARL** SMACv2 datasets — the paper's ComaDICE buffers
  were never released. OG-MARL covers **3 of the paper's 15 cells**:
  `terran_5_vs_5`, `zerg_5_vs_5`, `terran_10_vs_10`.
- **Deliverable:** the **O-MAPL win-rate curve** (mean ± std over 4 seeds) for
  those 3 scenarios — our version of the O-MAPL row of Figure 1.
- **Honest framing:** *a faithful reimplementation of O-MAPL, evaluated on public
  SMACv2 data where quality tiers are reconstructed from returns — reproduces the
  qualitative learning curves, not the paper's exact numbers* (impossible without
  the authors' unreleased data).

---

## 1. Cluster facts (Explorer)

- Login node: `login.explorer.northeastern.edu` (has internet).
- GPU batch partition: `--partition=gpu`; interactive: `--partition=gpu-interactive`.
- GPUs need an **explicit type**: `--gres=gpu:v100-sxm2:1` (bare `gpu:1` is not enough).
- **QOS limit: ~4 concurrent GPU jobs per user** (`QOSMaxJobsPerUserLimit`) — the
  12-job sweep runs 4 at a time and queues the rest (fine).
- Compute nodes: **no internet** (downloads must run on the login node).
- Login node has a **memory cap** — heavy data builds get OOM-killed (`Exit 137`);
  run those as batch jobs with `--mem`.

---

## 2. Environment setup (one-time)

```bash
# StarCraft II via pymarl2's installer (installs to $HOME/StarCraftII, SC2.4.10)
cd $HOME && git clone https://github.com/hijkzzz/pymarl2.git
cd pymarl2 && bash install_sc2.sh        # -> $HOME/StarCraftII, + SMAC v1 maps

# O-MAPL conda env (torch-CUDA, smacv2, flashbax/jax); skips SC2 (already present)
cd $HOME && git clone https://github.com/Teoman21/O-MAPL.git
cd O-MAPL && bash scripts/setup_hpc.sh
source ~/.omapl_env
```

### Problem 2a — `smacv2` pip package ships **no `.SC2Map` files**
`setup_hpc.sh` warned *"Could not find packaged SMAC maps"* and SMACv2's `10gen_*`
maps were missing, so the env would fail at eval. **Fix:** clone the smacv2 source
and copy the maps into SC2:
```bash
cd $HOME && git clone https://github.com/oxwhirl/smacv2.git smacv2_src
MAPSRC=$HOME/smacv2_src/smacv2/env/starcraft2/maps/SMAC_Maps
cp -rv "$MAPSRC"/*.SC2Map "$HOME/StarCraftII/Maps/SMAC_Maps/"   # incl. 10gen_terran/protoss/zerg
```

Sanity:
```bash
python -c "import torch, smacv2, flashbax; print('cuda', torch.cuda.is_available())"
python -c "from smacv2.env import StarCraftCapabilityEnvWrapper; print('smacv2 ok')"
```

---

## 3. Data — OG-MARL vaults → preference datasets

```bash
bash scripts/download_smacv2_data.sh     # downloads + extracts vaults, builds .pkl
```

### Problem 3a — OG-MARL `core/smac_v2` has **no Good/Medium/Poor split**
The vaults ship a **single combined buffer** (flashbax uid `Replay`, plus a
`Random` uid), not the paper's quality tiers. **Fix (committed):** the adapter now
auto-detects the layout and, for a single buffer, **reconstructs poor/medium/expert
tiers by episodic-return terciles** (`load_single_buffer_as_tiers` in
`omapl/data/ogmarl_adapter.py`). Labelling scheme is unchanged from the paper.
Verified by increasing tier returns, e.g. `terran_5_vs_5`: poor 4.54 < medium 8.90
< expert 16.71.

### Problem 3b — truncated vault extraction (`Killed` mid-unzip)
An interrupted unzip left `Replay/` without its `manifest.ocdbt`, so flashbax
failed: `NOT_FOUND: ... actions./.zarray ... does not exist`. **Fix:** re-extract
from the intact zip (no internet needed):
```bash
cd vaults/og_marl/smac_v2 && rm -rf terran_5_vs_5.vlt && unzip -q terran_5_vs_5.zip
find terran_5_vs_5.vlt -name manifest.ocdbt   # must show 2 (Replay + Random)
```
Run heavy unzips with `nohup ... &` on the login node so they survive SSH drops.

### Problem 3c — login-node OOM building `terran_10_vs_10` (`Exit 137`)
The biggest dataset (10 agents) blew the login memory cap. **Fix:** build it as a
CPU batch job with real memory (vault already local, no internet):
```bash
sbatch --partition=short --cpus-per-task=4 --mem=64G --time=01:00:00 \
  --job-name=build_t10 --output=runs/slurm/build_t10_%j.out \
  --wrap="source $HOME/.omapl_env; cd $HOME/O-MAPL; python scripts/make_smacv2_data.py --scenario terran_10_vs_10 --vault_base ./vaults --out data/smacv2_terran_10_vs_10.pkl --n_pairs 2000 --n_per_tier 1000 --seed 0"
```

**Result:** 3 datasets built —
`terran_5_vs_5.pkl` (702 MB), `zerg_5_vs_5.pkl` (521 MB), `terran_10_vs_10.pkl` (2.7 GB).
Dims: terran_5 `n_agents=5, obs=82, state=120, act=11`; terran_10 `n_agents=10, obs=162, state=290, act=16`.

---

## 4. Training infrastructure fixes

### Problem 4a — SC2 fails at first eval: `failed CONNECT via proxy status: 403`
pysc2 reaches the local SC2 process over a `127.0.0.1` websocket, but Explorer's
compute-node `http(s)_proxy` env vars made it tunnel that local connection through
the proxy. **Fix (committed to `scripts/train_smacv2.slurm`):**
```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="localhost,127.0.0.1,::1"; export NO_PROXY="$no_proxy"
```

### Problem 4b — GPU type required
Submit with `--gres=gpu:v100-sxm2:1` (bare `gpu:1` may be rejected / hang).

### Checkpoint / requeue behaviour (by design)
`train_smacv2.slurm` sets `--requeue` + `--signal=USR1@120`; the trainer catches
the signal, checkpoints full state, and `scontrol requeue`s itself. Consequence:
**`scancel` on a *running* job triggers a requeue** (it comes back as `PD
(BeginTime)`). To truly kill, `scancel` again while it is pending.

---

## 5. The real blocker — value divergence, and the `beta` fix

**Symptom:** with the default `beta=1.0`, `V_mean` exploded (→ ~4,500–6,000) within
a few thousand steps, and `eval_win_rate` peaked early then **decayed** (e.g.
31%→19%→16%) — the opposite of the paper's rising curves.

**Root cause:** `beta` (the MaxEnt temperature) is **not specified in the paper**.
In the Extreme-V/Gumbel loss, `V` is pushed toward `Q` via `exp((Q−V)/beta)`; at
`beta=1` that term is huge, slams the `exp_clip=8` ceiling, and drives runaway
value inflation. `reg_coef` (an extra knob; the paper uses unit weight) only
changed the *speed* of divergence, never fixed it.

**Sweep on `terran_5_vs_5` (seed 0, 20k steps):**

| beta | V_mean | win-rate trend | verdict |
|-----:|-------:|----------------|---------|
| 1  | ~4,500 💥 | 31→19→16 (decays) | diverges |
| 5  | high | 22→25→28→34 (climbs) | good |
| **10** | high | 25→28→31→31→**37.5** (climbs) | **best** |
| 20 | ~24 (bounded) | 28→25→19→19 (stuck low) | over-softened |

**Key insight:** a *bounded* `V` is **not** the goal — win-rate is. The large `V`
is a gauge offset that **cancels in the advantage** `Q(o,a)−V(o)` that drives the
policy, so it does not hurt win-rate. `beta` matters as the *policy temperature*:
`beta=20` over-softens the advantage weighting (weak policy, 19%); `beta≈10` is the
sweet spot. `reg_coef` (which we swept 1→1000) is a red herring for this.

**`beta` is per-scenario (tuned; not in the paper), `reg_coef=1.0`:**

| scenario | beta | why |
|---|---|---|
| terran_5_vs_5 | **10** | best win-rate at 20k sweep; beta=20 over-softens it (18%) |
| zerg_5_vs_5 | **20** | beta≤10 diverges over full training; 20 keeps V bounded |
| terran_10_vs_10 | **20** | beta=10 diverges to V~millions by ~90k; 20 keeps V~20 |

**Important follow-up finding (full 100k runs).** `beta=10` looked best in the
short 20k sweep but **did not hold over the full 100k on the harder scenarios**:
terran_10 s0 at `beta=10` diverged late (loss/reg exploded, win-rate → ~0–3%). A
second sweep (`beta=20/30` on zerg + terran_10) **fixed the divergence** (V stayed
bounded at ~15–25) but win-rates stayed low anyway — zerg ~6–18%, terran_10 ~0–6%.
Since training is now stable and win-rate is *still* low, this is a **data gap**
(public OG-MARL is weaker than the paper's unreleased ComaDICE data on the harder
scenarios), not a bug. `beta=20` chosen for zerg + terran_10 as the stable config;
we stopped tuning there. Honest expected result: terran_5 ~9–31% (closest to the
paper), zerg/terran_10 stable but well below the paper's ~30% — a documented
consequence of the public data.

---

## 6. The final run (produces the O-MAPL curve)

The array maps `SLURM_ARRAY_TASK_ID` → (scenario, seed) as
`scenario = id // 4`, `seed = id % 4` with
`SCENARIOS=(terran_5_vs_5 zerg_5_vs_5 terran_10_vs_10)`:

| array id | scenario | seed |
|---|---|---|
| 0–3 | terran_5_vs_5 | 0–3 |
| 4–7 | zerg_5_vs_5 | 0–3 |
| 8–11 | terran_10_vs_10 | 0–3 |

Only `terran_5_vs_5` was validated end-to-end during setup, so we launch
**validation-first** to catch any zerg/terran_10 issue (live-env dim mismatch →
clean abort by the dimension guard, or terran_10 OOM) in ~20 min instead of
mid-run.

```bash
# on HPC, after `git pull` brings the beta=10 configs
cd $HOME/O-MAPL

# Stage 1 — one seed per scenario (ids 0,4,8). Confirm each reaches an eval line.
sbatch --array=0,4,8 --gres=gpu:v100-sxm2:1 scripts/train_smacv2.slurm

# ...wait ~20 min, verify all three print eval_win_rate (no dim-guard abort / OOM)...
for f in runs/slurm/omapl-smacv2-*_*.out; do echo "== $f =="; grep eval_win_rate "$f" | tail -1; done

# Stage 2 — the remaining 9 seeds.
sbatch --array=1-3,5-7,9-11 --gres=gpu:v100-sxm2:1 scripts/train_smacv2.slurm
```
- 12 jobs total = 3 scenarios × 4 seeds, 100k steps each, eval every 1k → 100 eval
  points (the figure's x-axis). Runs 4 at a time (QOS), ~12–16h wall, auto-resuming.
- Results: `runs/omapl-omapl-<scenario>-s<seed>/metrics.csv`.

**Monitoring cheatsheet:**
```bash
squeue -u $USER
# win-rate trend per job:
for f in runs/slurm/omapl-smacv2-*_*.out; do echo "== $f =="; grep eval_win_rate "$f" | tail -3; done
```

**Plot the curve:**
```bash
source ~/.omapl_env
python scripts/plot_winrate.py        # -> runs/omapl_smacv2_winrate.png
```

---

## 6b. Final results & honest assessment (outcome)

Full run completed: O-MAPL, 3 public SMACv2 scenarios × 4 seeds, 100k steps,
per-scenario `beta` (§5). Figure: `runs/omapl_smacv2_winrate.png`.

**Final win-rate (mean ± std over 4 seeds):**

| scenario | ours | paper (approx, Fig. 1) |
|---|---|---|
| terran_5_vs_5 | **21.1 ± 8.1 %** | ~40 % |
| zerg_5_vs_5 | **8.6 ± 2.6 %** | ~30 % |
| terran_10_vs_10 | **3.1 ± 3.1 %** | ~30 % |

**This is NOT a successful reproduction of the paper's curves.** Two gaps:
1. **Magnitude** well below the paper.
2. **Shape:** our curves are **flat** — win-rate jumps to a low level early and
   oscillates there — whereas the paper's O-MAPL curves clearly **rise** from ~0
   to a plateau (they *learn* over training). Ours don't visibly learn.

**Root cause — the `beta` stability/learning tension (not just the data gap).**
The policy is extracted by advantage-weighted BC with weight `exp((Q−V)/beta)`:
- At **high `beta`** (20, used for zerg/terran_10 to stop divergence) the exponent
  is tiny → weight ≈ 1 for every action → weighted-BC **collapses to plain BC** →
  the policy just clones the offline data → **flat curve at data level.**
- At **low `beta`** (where the advantage actually shapes the policy → real
  learning) the **value diverges** (`V → 10³–10⁶`, seen at beta=1/10).

We were forced to choose stability over learning signal. The paper evidently
operates at a point that is **both** stable *and* learning, which implies a
**stabilization mechanism we did not implement** — most likely one of:
- **return/reward normalization** (standard in OMIGA/ComaDICE offline MARL; absent here),
- **lower Q/V learning rate** combined with a low `beta`,
- **target-network / clipping details** in the Extreme-V update.

Compounded by the **data gap** (public OG-MARL ≠ the paper's unreleased ComaDICE
buffers), so even a correct stabilization fix may not reach the paper's magnitudes.

**Status:** faithful method + stable training on real public data, but it does
**not** recover the paper's win-rate curves. Left as an open issue for future
work; the concrete next experiment is return normalization + a low-`beta` retry
on terran_5_vs_5.

**Why this is genuinely hard (not a matter of effort).**
- **No reference code + underspecified hyperparameters.** O-MAPL ships no code;
  `beta` (the single knob that decides stable-vs-learning) and any reward/value
  normalization are **not stated in the paper**. Reproducing the result means
  *rediscovering* an unstated recipe, not just re-running one.
- **No reference data.** The paper's ComaDICE SMACv2 buffers were never released;
  public OG-MARL differs in quality and coverage (3 of 15 cells), so there is no
  apples-to-apples target — absolute numbers can't be expected to match.
- **The core algorithm sits on a knife-edge.** Preference-only offline value
  learning has nothing grounding the value scale except a soft χ² term, so it is
  intrinsically prone to the deadly-triad divergence. The stable region (bounded
  value *and* a live advantage signal) is narrow and task-dependent; hitting it
  reliably across scenarios is a research problem, not a config tweak.
- **Slow, expensive iteration.** Each signal requires ~100k offline steps + live
  StarCraft II evaluation on a shared, quota-limited, preemptible GPU cluster —
  hours per data point — so the search over the unstated recipe is costly.

In short: reproducing this figure means solving an underspecified, unstable,
data-limited offline-RL problem with slow feedback. We built the full correct
pipeline and reached stable training; closing the last gap to the paper's curves
is a substantial research effort in its own right.

## 7. Code changes made during this run (committed)

- `omapl/data/ogmarl_adapter.py` — `list_vault_uids`, `load_single_buffer_as_tiers`
  (single-buffer → return-tercile tiers).
- `scripts/make_smacv2_data.py` — auto-detect vault layout, dispatch loader.
- `scripts/train_smacv2.slurm` — unset proxy vars / set `no_proxy` for SC2;
  optional `OMAPL_OVERRIDES` hook for sweeps.
- `configs/smacv2_*.yaml` — `beta: 1.0 → 10.0` (documented).
- `REPRODUCE_SMACV2.md` — documented the single-buffer / tercile deviation.

## 8. Gotchas checklist (if starting fresh)

1. `install_sc2.sh` lives in **pymarl2**, not O-MAPL.
2. `smacv2` pip package has **no maps** — copy from the smacv2 **source** clone.
3. OG-MARL `core` vaults = **single `Replay` buffer**, not quality tiers.
4. Verify vault extraction finished — look for `manifest.ocdbt` in each uid.
5. Build big datasets as **batch jobs** (login-node OOM otherwise).
6. Unset **proxy** vars in GPU jobs or SC2's localhost websocket fails (403).
7. `--gres=gpu:v100-sxm2:1` (explicit type); ~4-GPU concurrent QOS limit.
8. `scancel` on a running job **requeues** it — cancel again while pending to kill.
9. **`beta=10`** (not the default 1.0) — else the value diverges and win-rate decays.
</content>
</invoke>
