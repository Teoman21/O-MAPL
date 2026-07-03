# Reproducing the O-MAPL SMACv2 win-rate curves

This is the end-to-end runbook for reproducing O-MAPL's SMACv2 evaluation
curves (the per-map win-rate-vs-eval-step figure) on a SLURM GPU cluster —
written for **Northeastern's Discovery** cluster, but portable to any
Linux + SLURM + GPU machine.

## What this reproduces (and the honest caveat)

The paper's figure spans **15 cells** (Protoss / Terran / Zerg × 5 unit counts).
Those exact offline datasets come from **ComaDICE**, which **was never publicly
released** — neither O-MAPL nor ComaDICE ships code or data. The closest public
SMACv2 offline data is **OG-MARL** (InstaDeep, on Hugging Face), which covers
**3 of the 15 cells**:

| scenario          | available |
|-------------------|-----------|
| `terran_5_vs_5`   | ✅ Good / Medium / Poor |
| `zerg_5_vs_5`     | ✅ Good / Medium / Poor |
| `terran_10_vs_10` | ✅ Good / Medium / Poor |

So this reproduces the **method and curve shape** on real public SMACv2 data for
those 3 cells. Because OG-MARL ≠ the ComaDICE buffers, **absolute win rates will
not match the paper's numbers** — only the qualitative learning behaviour. To
match the paper exactly you'd need the ComaDICE datasets from the authors, or to
regenerate them with MAPPO (see "Going further").

The O-MAPL algorithm, the linear non-negative mixer, the three alternating
updates, discrete action masking, agent-ids, and all Table 6 hyperparameters are
already implemented and unit-tested — see `NOTES.md`.

## Step 0 — get the repo onto Discovery

```bash
# from a Discovery login node
git clone <your-fork-or-copy> O-MAPL && cd O-MAPL   # or rsync this directory up
```

## Step 1 — one-time environment setup

```bash
bash scripts/setup_hpc.sh          # conda env + torch(CUDA) + smacv2 + jax/flashbax + StarCraft II
source ~/.omapl_env                # activates env, exports SC2PATH
# sanity:
python -c "import torch, smacv2, flashbax; print('cuda', torch.cuda.is_available())"
python -c "from smacv2.env import StarCraftCapabilityEnvWrapper; print('smacv2 ok')"
```

`setup_hpc.sh` installs StarCraft II 4.10 headless (the build SMAC targets) and
copies the SMAC maps into `$SC2PATH/Maps`. **Adjust the `module load` lines and
the PyTorch CUDA wheel** to match Discovery's current modules (`module avail
cuda`, `module avail anaconda3`).

## Step 2 — download data + build preference datasets

```bash
bash scripts/download_smacv2_data.sh
# -> ./vaults/og_marl/smac_v2/<scenario>.vlt   (downloaded)
# -> data/smacv2_<scenario>.pkl                (2000 preference pairs each)
```

This pulls the OG-MARL vaults from Hugging Face, slices each Good/Medium/Poor
split into per-episode trajectories (`omapl/data/ogmarl_adapter.py`), and builds
rule-based preference pairs (cross-quality pairs labelled by tier, same-tier by
return — the paper's poor/medium/expert scheme). 2000 pairs/task per Table 3.

## Step 3 — train (3 scenarios × 4 seeds)

```bash
mkdir -p runs/slurm
sbatch scripts/train_smacv2.slurm     # SLURM array 0-11 = 3 scenarios x 4 seeds
```

Each run is 100k steps with evaluation every 1000 steps (→ 100 eval points,
matching the figure's x-axis), 32 episodes/eval, against the live SMACv2 env.
All hyperparameters come from Table 6 (`configs/smacv2_*.yaml`). Results land in
`runs/omapl-omapl-<scenario>-s<seed>/metrics.csv`.

**Preemption / time-limit safety (checkpointing).** GPU jobs get preempted or
hit the partition wall clock. Every run writes a *full-state* checkpoint
(`runs/<exp>/checkpoint.pt` — weights, optimizer moments, RNG, step counter)
every `ckpt_every` steps. The SLURM script sets `--requeue` and
`--signal=USR1@120`: 120 s before the limit (or on preemption) the trainer saves
a checkpoint and `scontrol requeue`s itself; SLURM re-runs the script and
`python -m omapl.train` auto-resumes bit-identically (`cfg.resume=true`). So a
killed job is **fully autonomous** — no manual resubmission, and resuming is
verified exact by `tests/test_checkpoint.py`. To force a fresh run, delete the
run dir (or pass `resume=false`).

A dimension guard in `omapl/train.py` aborts with a clear message if the live
SMACv2 env's obs/state/action dims don't match the dataset — the most likely
failure if a SMACv2 version ships different obs flags than OG-MARL used.

## Step 4 — plot the curves

```bash
source ~/.omapl_env
python scripts/plot_winrate.py        # -> runs/omapl_smacv2_winrate.png
```

Produces one panel per scenario, win-rate mean ± std band over the 4 seeds
(O-MAPL in red), plus a printed final-win-rate summary table.

## Going further — the full 15-cell figure

Not possible from public data. Options:
1. **Email the authors** (Bui / Mai / Nguyen, SMU) for the ComaDICE SMACv2
   buffers + O-MAPL poor/expert splits + preference labels.
2. **Regenerate the datasets**: train MAPPO on each SMACv2 map (~10M steps) per
   ComaDICE's procedure, sample trajectories at poor/medium/expert quality, then
   label with `omapl/data/generate_preferences.py` (rule-based, or the exact
   Table-5 GPT-4o prompt via `annotate_with_llm`). Heavy GPU cost; covers all 15
   cells. The env wrapper already supports all three races and arbitrary unit
   counts (`omapl/envs/smac_wrapper.py`), so only the data step is missing.
```
