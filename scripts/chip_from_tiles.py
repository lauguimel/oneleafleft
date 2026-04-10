"""Chip local GeoTIFFs into HDF5 training data.

Reads full-extent GeoTIFFs exported from GEE (scripts/export_tiles_gee.py)
and cuts 64×64 chips centred on each point in the chip index.

Much faster than per-chip GEE extraction (~minutes vs ~days).

Input:
  - data/tiles/hls_{year}.tif — annual HLS composites (8 bands: B2-B7, NDVI, NBR)
  - data/tiles/srtm.tif — elevation + slope (2 bands)
  - data/tiles/hansen_treecover2000.tif — treecover (1 band)
  - data/tiles/hansen_lossyear.tif — lossyear (1 band)
  - data/chip_index.parquet — chip locations + metadata

Output:
  - data/chips/{split}_chips.h5 — HDF5 with images + masks

Usage:
    conda run -n deforest python scripts/chip_from_tiles.py
    conda run -n deforest python scripts/chip_from_tiles.py --tiles-dir data/tiles
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

CHIP_PX = 64
FEATURE_WINDOW = 5

# Band layout in HDF5: 6 HLS + NDVI + NBR + elevation + slope + treecover2000 = 11
N_HLS = 8     # B2, B3, B4, B5, B6, B7, NDVI, NBR
N_STATIC = 3  # elevation, slope, treecover2000
N_CHANNELS = N_HLS + N_STATIC


def lonlat_to_pixel(transform, lon, lat):
    """Convert lon/lat to pixel row/col using rasterio transform."""
    col, row = ~transform * (lon, lat)
    return int(round(row)), int(round(col))


def read_chip(src, row_center, col_center, size=CHIP_PX):
    """Read a size×size chip centred on (row_center, col_center)."""
    half = size // 2
    row_off = row_center - half
    col_off = col_center - half

    # Handle boundary: pad with nodata if needed
    win = Window(col_off, row_off, size, size)
    data = src.read(window=win, boundless=True, fill_value=0)
    return data.astype(np.float32)


def build_loss_mask(lossyear_chip, pred_year):
    """Binary mask: 1 where lossyear in [pred_year, pred_year+2]."""
    codes = [pred_year - 2000 + i for i in range(3)]
    mask = np.zeros_like(lossyear_chip, dtype=np.uint8)
    for code in codes:
        mask |= (lossyear_chip == code).astype(np.uint8)
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiles-dir", type=Path, default=DATA_DIR / "tiles")
    parser.add_argument("--index", type=Path, default=DATA_DIR / "chip_index.parquet")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "chips")
    args = parser.parse_args()

    tiles_dir = args.tiles_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load chip index
    print(f"[1] Loading chip index: {args.index}")
    df = pd.read_parquet(args.index)
    print(f"    {len(df):,} chips")

    # Determine which years we need
    all_years = set()
    for _, row in df.iterrows():
        for yr in range(int(row.feat_year_start), int(row.feat_year_end) + 1):
            all_years.add(yr)
    all_years = sorted(all_years)
    print(f"    Feature years needed: {all_years}")

    # Open rasters
    print(f"\n[2] Opening GeoTIFFs from {tiles_dir}")
    hls_srcs = {}
    for yr in all_years:
        path = tiles_dir / f"hls_{yr}.tif"
        if not path.exists():
            print(f"    WARNING: {path} not found, will fill with zeros")
            hls_srcs[yr] = None
        else:
            hls_srcs[yr] = rasterio.open(path)
            print(f"    hls_{yr}: {hls_srcs[yr].width}×{hls_srcs[yr].height}, {hls_srcs[yr].count} bands")

    srtm_path = tiles_dir / "srtm.tif"
    tc_path = tiles_dir / "hansen_treecover2000.tif"
    ly_path = tiles_dir / "hansen_lossyear.tif"

    srtm_src = rasterio.open(srtm_path) if srtm_path.exists() else None
    tc_src = rasterio.open(tc_path) if tc_path.exists() else None
    ly_src = rasterio.open(ly_path) if ly_path.exists() else None

    if srtm_src:
        print(f"    srtm: {srtm_src.count} bands")
    if tc_src:
        print(f"    treecover2000: {tc_src.count} bands")
    if ly_src:
        print(f"    lossyear: {ly_src.count} bands")

    # Use the first available HLS raster for coordinate transform
    ref_src = next((s for s in hls_srcs.values() if s is not None), ly_src)
    if ref_src is None:
        print("ERROR: No GeoTIFFs found!")
        return
    transform = ref_src.transform

    # Process by split
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name].reset_index(drop=True)
        n = len(split_df)
        if n == 0:
            continue

        out_path = out_dir / f"{split_name}_chips.h5"
        print(f"\n[3] Chipping {n:,} chips for split='{split_name}' → {out_path}")

        with h5py.File(out_path, "w") as f:
            f.create_dataset("images",
                shape=(n, FEATURE_WINDOW, N_CHANNELS, CHIP_PX, CHIP_PX),
                dtype=np.float32, chunks=(1, FEATURE_WINDOW, N_CHANNELS, CHIP_PX, CHIP_PX),
                compression="gzip", compression_opts=4)
            f.create_dataset("masks",
                shape=(n, CHIP_PX, CHIP_PX), dtype=np.uint8,
                chunks=(1, CHIP_PX, CHIP_PX), compression="gzip", compression_opts=4)
            f.create_dataset("pid", shape=(n,), dtype=np.int64)
            f.create_dataset("pred_year", shape=(n,), dtype=np.int16)
            f.create_dataset("lon", shape=(n,), dtype=np.float64)
            f.create_dataset("lat", shape=(n,), dtype=np.float64)

            n_ok = 0
            for i, row in split_df.iterrows():
                r, c = lonlat_to_pixel(transform, row.lon, row.lat)
                pred_yr = int(row.pred_year)
                feat_years = list(range(int(row.feat_year_start), int(row.feat_year_end) + 1))

                # Read static layers once
                static_chips = []
                if srtm_src:
                    srtm_chip = read_chip(srtm_src, r, c)  # (2, 64, 64)
                else:
                    srtm_chip = np.zeros((2, CHIP_PX, CHIP_PX), dtype=np.float32)

                if tc_src:
                    tc_chip = read_chip(tc_src, r, c)  # (1, 64, 64)
                else:
                    tc_chip = np.zeros((1, CHIP_PX, CHIP_PX), dtype=np.float32)

                static = np.concatenate([srtm_chip, tc_chip], axis=0)  # (3, 64, 64)

                # Read temporal HLS chips
                image_cube = np.zeros((FEATURE_WINDOW, N_CHANNELS, CHIP_PX, CHIP_PX), dtype=np.float32)
                for t_idx, yr in enumerate(feat_years[:FEATURE_WINDOW]):
                    src = hls_srcs.get(yr)
                    if src is not None:
                        hls_chip = read_chip(src, r, c)  # (8, 64, 64)
                    else:
                        hls_chip = np.zeros((N_HLS, CHIP_PX, CHIP_PX), dtype=np.float32)

                    # Stack: HLS bands + static
                    image_cube[t_idx, :N_HLS] = hls_chip
                    image_cube[t_idx, N_HLS:] = static

                # Loss mask
                if ly_src:
                    ly_chip = read_chip(ly_src, r, c)[0]  # (64, 64)
                    mask = build_loss_mask(ly_chip, pred_yr)
                else:
                    mask = np.zeros((CHIP_PX, CHIP_PX), dtype=np.uint8)

                # NaN → 0
                image_cube = np.nan_to_num(image_cube, nan=0.0)

                f["images"][i] = image_cube
                f["masks"][i] = mask
                f["pid"][i] = row.pid
                f["pred_year"][i] = pred_yr
                f["lon"][i] = row.lon
                f["lat"][i] = row.lat
                n_ok += 1

                if (n_ok) % 1000 == 0 or n_ok == n:
                    print(f"    {n_ok}/{n} chips done")

        print(f"    Saved: {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")

    # Cleanup
    for src in hls_srcs.values():
        if src is not None:
            src.close()
    for src in [srtm_src, tc_src, ly_src]:
        if src is not None:
            src.close()

    print("\nDone! Ready for training.")


if __name__ == "__main__":
    main()
