#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-time environment setup for reproducing O-MAPL on SMACv2, targeting the
# Northeastern Discovery cluster (SLURM + GPU + RHEL/Rocky Linux). Run this on a
# LOGIN node (or an interactive `srun` shell). It builds a conda env, installs
# PyTorch + smacv2 + the OG-MARL data deps, and installs StarCraft II headless
# with the SMACv2 maps.
#
# Discovery notes (verify against current docs.rc.northeastern.edu):
#   * module names/versions change — adjust `module load` lines if they fail.
#   * GPUs live on the `gpu` partition; SC2 evaluation itself is CPU.
#   * Heavy downloads: run on a node with internet (login nodes have it).
#
# Usage:
#   bash scripts/setup_hpc.sh
#   # then: source ~/.omapl_env   (exports SC2PATH and activates the env)
# ---------------------------------------------------------------------------
set -euo pipefail

ENV_NAME="${ENV_NAME:-omapl}"
PY_VER="${PY_VER:-3.10}"
SC2DIR="${SC2DIR:-$HOME/StarCraftII}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Repo: $REPO_DIR"
echo "==> Conda env: $ENV_NAME (python $PY_VER)"
echo "==> SC2 install dir: $SC2DIR"

# --- 1. Conda environment --------------------------------------------------
# On Discovery: `module load anaconda3/2022.05` (or miniconda3). If conda is
# already on PATH this is a no-op.
module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^${ENV_NAME} "; then
  conda create -y -n "$ENV_NAME" python="$PY_VER"
fi
conda activate "$ENV_NAME"

# --- 2. Python deps --------------------------------------------------------
python -m pip install --upgrade pip wheel

# PyTorch: install a CUDA build matching the cluster's CUDA. Check `nvidia-smi`
# / `module avail cuda`. Example for CUDA 12.1; adjust the index URL as needed.
if python -c "import torch" 2>/dev/null; then
  echo "==> torch already installed"
else
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu121 \
    || python -m pip install torch   # fallback (CPU) — replace with the right CUDA wheel
fi

# Base repo deps (numpy, pyyaml) + plotting + h5py.
python -m pip install -r "$REPO_DIR/requirements.txt"
python -m pip install matplotlib pandas h5py

# OG-MARL data loading: flashbax vaults need jax (CPU is fine for conversion).
python -m pip install "jax[cpu]" flashbax

# SMACv2 (includes the v1 maps + 10gen procedural maps).
python -m pip install "git+https://github.com/oxwhirl/smacv2.git"

# --- 3. StarCraft II (headless Linux) + SMAC maps --------------------------
# Uses the SC2 4.10 Linux package that SMAC targets. The zip password is the
# Blizzard AI EULA acknowledgement string (public, documented by DeepMind).
if [ ! -d "$SC2DIR" ]; then
  echo "==> Downloading StarCraft II 4.10 (Linux headless)..."
  TMP_ZIP="$HOME/SC2.4.10.zip"
  wget -q --show-progress -O "$TMP_ZIP" \
    "http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip"
  echo "==> Unzipping to $HOME (password = iagreetotheeula)"
  unzip -P iagreetotheeula -q "$TMP_ZIP" -d "$HOME"
  rm -f "$TMP_ZIP"
else
  echo "==> SC2 already present at $SC2DIR"
fi

# SMAC maps: smacv2 ships SMAC_Maps + the 10gen maps under its package dir.
SMACV2_MAPS="$(python -c 'import os,smacv2; print(os.path.join(os.path.dirname(smacv2.__file__),"env","starcraft2","maps","SMAC_Maps"))')"
mkdir -p "$SC2DIR/Maps"
if [ -d "$SMACV2_MAPS" ]; then
  echo "==> Copying SMAC maps from $SMACV2_MAPS"
  cp -r "$SMACV2_MAPS" "$SC2DIR/Maps/"
else
  echo "[warn] Could not find packaged SMAC maps. Copy SMAC_Maps into $SC2DIR/Maps manually."
fi

# --- 4. Persist environment ------------------------------------------------
cat > "$HOME/.omapl_env" <<EOF
# Source this before running O-MAPL jobs.
module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME
export SC2PATH="$SC2DIR"
EOF

echo
echo "==> Done. To use:"
echo "    source ~/.omapl_env"
echo "    python -c 'import torch, smacv2, flashbax; print(\"ok\", torch.cuda.is_available())'"
echo "    python -c 'from smacv2.env import StarCraftCapabilityEnvWrapper; print(\"smacv2 ok\")'"
