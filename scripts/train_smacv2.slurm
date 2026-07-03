#!/usr/bin/env bash
#SBATCH --job-name=omapl-smacv2
#SBATCH --partition=gpu              # Northeastern Explorer GPU partition
#SBATCH --gres=gpu:1                 # 1 GPU (type it if needed, e.g. gpu:v100-sxm2:1 / gpu:a100:1)
#SBATCH --cpus-per-task=4            # SC2 eval is CPU-bound; a few cores help
#SBATCH --mem=32G
#SBATCH --time=08:00:00             # wall limit; training auto-resumes, so this can be modest
#SBATCH --requeue                    # allow SLURM to requeue on preemption
#SBATCH --signal=USR1@120            # send SIGUSR1 to the job step 120s before the time limit
#SBATCH --open-mode=append           # keep logs across requeues
#SBATCH --array=0-11                 # 3 scenarios x 4 seeds
#SBATCH --output=runs/slurm/%x-%A_%a.out
#SBATCH --error=runs/slurm/%x-%A_%a.err
# ---------------------------------------------------------------------------
# Train O-MAPL on the three public SMACv2 scenarios across 4 seeds (Table 6).
# Each array task = one (scenario, seed). Submit from the repo root:
#
#   mkdir -p runs/slurm
#   sbatch scripts/train_smacv2.slurm
#
# Preemption / time-limit safety:
#   * Training checkpoints its FULL state (weights + optimizer + RNG + step) to
#     runs/<exp>/checkpoint.pt every `ckpt_every` steps (omapl/utils/checkpoint.py).
#   * On SIGUSR1 (sent 120s before the wall limit) or preemption SIGTERM, it
#     saves a checkpoint and `scontrol requeue`s this job.
#   * With --requeue, SLURM re-runs this script; `python -m omapl.train` then
#     auto-resumes from checkpoint.pt (cfg.resume=true). Fully autonomous — you
#     do not need to resubmit by hand after a preemption or a hit time limit.
#
# Prereqs: setup_hpc.sh + download_smacv2_data.sh have been run.
# Results -> runs/omapl-omapl-<scenario>-s<seed>/{metrics.csv,checkpoint.pt,best.pt}
# ---------------------------------------------------------------------------
set -euo pipefail

source "$HOME/.omapl_env"

SCENARIOS=(terran_5_vs_5 zerg_5_vs_5 terran_10_vs_10)
SEEDS=(0 1 2 3)

n_seeds=${#SEEDS[@]}
sidx=$(( SLURM_ARRAY_TASK_ID / n_seeds ))
seed=${SEEDS[$(( SLURM_ARRAY_TASK_ID % n_seeds ))]}
scenario=${SCENARIOS[$sidx]}

echo "==> Task $SLURM_ARRAY_TASK_ID : scenario=$scenario seed=$seed (restart count: ${SLURM_RESTART_COUNT:-0})"
echo "==> SC2PATH=$SC2PATH  CUDA=$(python -c 'import torch;print(torch.cuda.is_available())')"

cd "$SLURM_SUBMIT_DIR"
# --signal delivers SIGUSR1 to the step below; the trainer's handler checkpoints
# and requeues. `srun` propagates the signal to the python process.
srun --unbuffered python -m omapl.train \
  --config "configs/smacv2_${scenario}.yaml" \
  seed="$seed" device=cuda
