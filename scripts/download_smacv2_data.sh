#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Download the public OG-MARL SMACv2 offline datasets (the three scenarios that
# exist) and build O-MAPL preference datasets from them.
#
#   Run AFTER setup_hpc.sh, with the env active:  source ~/.omapl_env
#   Usage:  bash scripts/download_smacv2_data.sh
#
# Vaults are flashbax format; quality splits Good/Medium/Poor are read by uid.
# Final layout (what flashbax expects):
#   ./vaults/og_marl/smac_v2/<scenario>.vlt
# Datasets are written to data/smacv2_<scenario>.pkl (2000 preference pairs each).
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

VAULT_BASE="${VAULT_BASE:-./vaults}"
DEST="$VAULT_BASE/og_marl/smac_v2"
HF="https://huggingface.co/datasets/InstaDeepAI/og-marl/resolve/main/core/smac_v2"
SCENARIOS=(terran_5_vs_5 zerg_5_vs_5 terran_10_vs_10)

mkdir -p "$DEST"
for s in "${SCENARIOS[@]}"; do
  if [ -d "$DEST/$s.vlt" ]; then
    echo "==> $s.vlt already present, skipping download"
  else
    echo "==> Downloading $s ..."
    wget -q --show-progress -O "$DEST/$s.zip" "$HF/$s.zip"
    echo "==> Unzipping $s ..."
    unzip -q -o "$DEST/$s.zip" -d "$DEST"
    rm -f "$DEST/$s.zip"
    # Normalise: ensure the vault dir is exactly $DEST/$s.vlt
    if [ ! -d "$DEST/$s.vlt" ]; then
      found="$(find "$DEST" -maxdepth 2 -type d -name "$s*vlt" | head -1 || true)"
      [ -n "$found" ] && [ "$found" != "$DEST/$s.vlt" ] && mv "$found" "$DEST/$s.vlt"
    fi
  fi
done

echo
echo "==> Building preference datasets (2000 pairs/scenario)..."
for s in "${SCENARIOS[@]}"; do
  python scripts/make_smacv2_data.py --scenario "$s" --vault_base "$VAULT_BASE" \
    --out "data/smacv2_$s.pkl" --n_pairs 2000 --n_per_tier 1000 --seed 0
done

echo
echo "==> Done. Datasets:"
ls -la data/smacv2_*.pkl
