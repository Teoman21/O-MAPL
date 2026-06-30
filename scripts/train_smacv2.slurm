#!/usr/bin/env bash
#SBATCH --job-name=omapl-smacv2
#SBATCH --partition=gpu              # Discovery GPU partition (adjust if needed)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
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
# Prereqs: setup_hpc.sh + download_smacv2_data.sh have been run.
# Results -> runs/omapl-omapl-<scenario>-s<seed>/metrics.csv  (then plot_winrate.py)
# ---------------------------------------------------------------------------
set -euo pipefail

source "$HOME/.omapl_env"

SCENARIOS=(terran_5_vs_5 zerg_5_vs_5 terran_10_vs_10)
SEEDS=(0 1 2 3)

n_seeds=${#SEEDS[@]}
sidx=$(( SLURM_ARRAY_TASK_ID / n_seeds ))
seed=${SEEDS[$(( SLURM_ARRAY_TASK_ID % n_seeds ))]}
scenario=${SCENARIOS[$sidx]}

echo "==> Task $SLURM_ARRAY_TASK_ID : scenario=$scenario seed=$seed"
echo "==> SC2PATH=$SC2PATH  CUDA=$(python -c 'import torch;print(torch.cuda.is_available())')"

cd "$SLURM_SUBMIT_DIR"
srun python -m omapl.train \
  --config "configs/smacv2_${scenario}.yaml" \
  seed="$seed" device=cuda
