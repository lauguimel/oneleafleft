#!/bin/bash
# Download exported GeoTIFFs from Google Drive and chip into HDF5.
#
# Prerequisites:
#   - All GEE exports completed (check: python scripts/export_tiles_gee.py --check)
#   - gdown installed (pip install gdown)
#
# Usage:
#   bash scripts/download_tiles.sh

set -e
cd "$(dirname "$0")/.."

TILES_DIR="data/tiles"
mkdir -p "$TILES_DIR"

echo "=== Step 1: Download from Google Drive ==="
echo "The GEE exports land in your Drive folder 'deforest_tiles/'."
echo ""
echo "Option A — gdown (simple, if you know the folder ID):"
echo "  gdown --folder https://drive.google.com/drive/folders/<FOLDER_ID> -O $TILES_DIR"
echo ""
echo "Option B — rclone (robust, handles large files):"
echo "  rclone copy gdrive:deforest_tiles/ $TILES_DIR/"
echo ""
echo "Option C — Manual: open Drive, select all files, download ZIP, unzip to $TILES_DIR/"
echo ""

# Check if tiles are already downloaded
EXPECTED_FILES=(
    "hls_2014.tif" "hls_2015.tif" "hls_2016.tif" "hls_2017.tif"
    "hls_2018.tif" "hls_2019.tif" "hls_2020.tif" "hls_2021.tif"
    "hansen_lossyear.tif" "hansen_treecover2000.tif" "srtm.tif"
)

MISSING=0
for f in "${EXPECTED_FILES[@]}"; do
    if [ ! -f "$TILES_DIR/$f" ]; then
        echo "  MISSING: $TILES_DIR/$f"
        MISSING=$((MISSING + 1))
    fi
done

if [ "$MISSING" -gt 0 ]; then
    echo ""
    echo "$MISSING files missing. Download them first, then re-run this script."
    echo ""
    echo "Trying gdown (will prompt for auth if needed)..."
    conda run -n deforest gdown --folder --remaining-ok \
        "https://drive.google.com/drive/folders/deforest_tiles" \
        -O "$TILES_DIR" 2>/dev/null || {
        echo ""
        echo "gdown failed (folder ID needed). Use manual download or rclone."
        echo "After downloading, re-run: bash scripts/download_tiles.sh"
        exit 1
    }
fi

echo ""
echo "=== Step 2: Verify tiles ==="
for f in "${EXPECTED_FILES[@]}"; do
    if [ -f "$TILES_DIR/$f" ]; then
        SIZE=$(du -h "$TILES_DIR/$f" | cut -f1)
        echo "  OK: $f ($SIZE)"
    else
        echo "  MISSING: $f"
    fi
done

echo ""
echo "=== Step 3: Chip into HDF5 ==="
conda run -n deforest python scripts/chip_from_tiles.py --tiles-dir "$TILES_DIR"

echo ""
echo "=== Step 4: Verify HDF5 ==="
conda run -n deforest python -c "
import h5py, numpy as np
for split in ['train', 'val', 'test']:
    path = 'data/chips/${split}_chips.h5'
    try:
        with h5py.File(path, 'r') as f:
            n = f['images'].shape[0]
            mask_pos = f['masks'][:].sum()
            mask_total = f['masks'][:].size
            print(f'  {split}: {n} chips, {mask_pos}/{mask_total} positive pixels ({100*mask_pos/mask_total:.1f}%)')
    except FileNotFoundError:
        print(f'  {split}: NOT FOUND')
"

echo ""
echo "=== Done! ==="
echo "Next steps:"
echo "  1. Setup Aqua (once): bash aqua/setup_env.sh"
echo "  2. Deploy + train:    bash aqua/deploy.sh"
