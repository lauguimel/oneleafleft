#!/bin/bash
# Full pipeline on Aqua: download tiles + chip + train
#
# Run this ONCE on Aqua after:
#   1. GEE exports are complete (check on https://code.earthengine.google.com/tasks)
#   2. Code is synced: rsync from local (see below)
#   3. Conda env 'deforest' is set up (bash aqua/setup_env.sh)
#   4. GEE + rclone auth done (see setup instructions below)
#
# === INITIAL SETUP (one-time, on Aqua login node) ===
#
#   # 1. Sync code from local laptop
#   # (on LOCAL machine):
#   rsync -avz --delete \
#       --exclude 'data/' --exclude 'models/' --exclude '.git/' \
#       --exclude '__pycache__/' --exclude 'logs/' --exclude 'results/' \
#       ~/Documents/Recherche/Deforestation/ \
#       maitreje@aqua.qut.edu.au:~/Deforestation/
#
#   # 2. SSH to Aqua
#   ssh maitreje@aqua.qut.edu.au
#
#   # 3. Setup rclone for Google Drive (one-time)
#   #    If rclone not available: pip install gdown
#   rclone config
#   #    → New remote → name: gdrive → type: drive → follow prompts
#
#   # NO GEE needed on Aqua — check export status from your browser:
#   #   https://code.earthengine.google.com/tasks
#
# === USAGE ===
#
#   # Interactive node (for download + chip — no GPU needed):
#   qsub -I -S /bin/bash -l select=1:ncpus=4:mem=32gb -l walltime=04:00:00
#   cd ~/Deforestation
#   bash aqua/full_pipeline.sh
#
#   # Then submit GPU training:
#   qsub aqua/train_prithvi.pbs

set -e
cd "${PBS_O_WORKDIR:-$HOME/Deforestation}"

echo "=============================================="
echo " Deforestation Pipeline — Aqua"
echo "=============================================="

# ── Conda ──────────────────────────────────────────
module load Miniconda3/24.9.2-0 || module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate deforest

TILES_DIR="data/tiles"
CHIPS_DIR="data/chips"
mkdir -p "$TILES_DIR" "$CHIPS_DIR" logs

# ── Step 1: Verify tiles are ready ─────────────────
echo ""
echo "[1/4] Checking if GEE exports have been downloaded..."
echo "  (Check export status at: https://code.earthengine.google.com/tasks)"

EXPECTED=(
    hls_2014 hls_2015 hls_2016 hls_2017
    hls_2018 hls_2019 hls_2020 hls_2021
    hansen_lossyear hansen_treecover2000 srtm
)

# ── Step 2: Download from Google Drive ─────────────
echo ""
echo "[2/4] Downloading tiles from Google Drive..."

if command -v rclone &> /dev/null; then
    echo "  Using rclone..."
    rclone copy gdrive:deforest_tiles/ "$TILES_DIR/" --progress
elif command -v gdown &> /dev/null; then
    echo "  Using gdown..."
    echo "  NOTE: you need the Drive folder URL. Find it at:"
    echo "  https://drive.google.com → deforest_tiles/ → share → copy link"
    echo ""
    read -p "  Paste the folder URL (or 'skip' if already downloaded): " FOLDER_URL
    if [ "$FOLDER_URL" != "skip" ]; then
        gdown --folder "$FOLDER_URL" -O "$TILES_DIR" --remaining-ok
    fi
else
    echo "  Neither rclone nor gdown found."
    echo "  Install: pip install gdown"
    echo "  Or: pip install rclone"
    exit 1
fi

# Verify downloads
echo ""
echo "  Verifying tiles..."
MISSING=0
for name in "${EXPECTED[@]}"; do
    if [ -f "$TILES_DIR/${name}.tif" ]; then
        SIZE=$(du -h "$TILES_DIR/${name}.tif" | cut -f1)
        echo "    OK: ${name}.tif ($SIZE)"
    else
        echo "    MISSING: ${name}.tif"
        MISSING=$((MISSING + 1))
    fi
done

if [ "$MISSING" -gt 0 ]; then
    echo ""
    echo "  ERROR: $MISSING files missing. Fix downloads and re-run."
    exit 1
fi

# ── Step 3: Chip into HDF5 ────────────────────────
echo ""
echo "[3/4] Chipping tiles into HDF5..."
python scripts/chip_from_tiles.py --tiles-dir "$TILES_DIR"

# ── Step 4: Verify ────────────────────────────────
echo ""
echo "[4/4] Verifying HDF5 chips..."
python -c "
import h5py, numpy as np
for split in ['train', 'val', 'test']:
    path = f'data/chips/{split}_chips.h5'
    try:
        with h5py.File(path, 'r') as f:
            n = f['images'].shape[0]
            shape = f['images'].shape
            mask_pos = f['masks'][:].sum()
            mask_total = f['masks'][:].size
            print(f'  {split}: {n} chips, shape={shape}, '
                  f'{mask_pos}/{mask_total} pos pixels ({100*mask_pos/mask_total:.1f}%)')
    except FileNotFoundError:
        print(f'  {split}: NOT FOUND')
"

echo ""
echo "=============================================="
echo " Pipeline complete!"
echo "=============================================="
echo ""
echo " Submit training:"
echo "   qsub aqua/train_prithvi.pbs"
echo ""
echo " After training, submit inference:"
echo "   qsub -v PRED_YEAR=2026 aqua/predict.pbs"
echo ""
