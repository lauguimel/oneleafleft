"""Fast chip extraction from pre-computed GEE Assets.

Reads HLS composites from GEE Assets (no on-the-fly computation),
extracts 64×64 chips via computePixels, and writes to HDF5.

Much faster than extract_chips_hls.py because:
- Composites are pre-computed (stored as Assets)
- computePixels just reads pixels, no median/cloud-mask computation
- Parallelized with ThreadPoolExecutor

Also extracts Hansen lossyear mask + SRTM + treecover2000 (static, fast).

Resumable: tracks progress in data/chips/_progress.json.

Usage:
    conda run -n deforest python scripts/extract_chips_fast.py
    conda run -n deforest python scripts/extract_chips_fast.py --workers 8 --test 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

DATA_DIR = PROJECT_DIR / "data"
CHIPS_DIR = DATA_DIR / "chips"

# ── Config ────────────────────────────────────────────────────────────────────
CHIP_PX = 64
CHIP_RES = 30
FEATURE_WINDOW = 5

ASSET_FOLDER = "projects/ee-guillaumemaitrejean/assets/deforest"
HANSEN_ASSET = "UMD/hansen/global_forest_change_2024_v1_12"
SRTM_ASSET = "USGS/SRTMGL1_003"

N_HLS_BANDS = 8   # B2-B7 + NDVI + NBR
N_STATIC = 3       # elevation, slope, treecover2000
N_CHANNELS = N_HLS_BANDS + N_STATIC

INDEX_PATH = DATA_DIR / "chip_index.parquet"
PROGRESS_FILE = CHIPS_DIR / "_progress_fast.json"


def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {}


def _save_progress(progress: dict):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress))


# ── GEE extraction ───────────────────────────────────────────────────────────

def _init_ee():
    import ee
    from data.gee_utils import init_gee
    init_gee()
    return ee


def _get_transform(lon, lat):
    """Affine transform for a chip centred at (lon, lat)."""
    px_deg = CHIP_RES / 111320.0
    half_deg = (CHIP_PX / 2) * px_deg
    return {
        "scaleX": px_deg, "shearX": 0,
        "translateX": lon - half_deg,
        "shearY": 0, "scaleY": -px_deg,
        "translateY": lat + half_deg,
    }


def _compute_chip(ee, image, lon, lat):
    """Extract a (C, H, W) numpy array via computePixels."""
    result = ee.data.computePixels({
        "expression": image,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": CHIP_PX, "height": CHIP_PX},
            "affineTransform": _get_transform(lon, lat),
            "crsCode": "EPSG:4326",
        },
    })
    bands = list(result.dtype.names)
    return np.stack([result[b].astype(np.float32) for b in bands], axis=0)


def extract_one_chip(ee, lon, lat, pred_year, hls_assets, static_img, lossyear_img):
    """Extract one complete chip: (T, C, H, W) image + (H, W) mask."""
    feat_years = list(range(pred_year - FEATURE_WINDOW, pred_year))

    # Static layers (once per chip)
    static_arr = _compute_chip(ee, static_img, lon, lat)  # (3, 64, 64)

    # Temporal HLS from pre-computed assets
    cube = np.zeros((FEATURE_WINDOW, N_CHANNELS, CHIP_PX, CHIP_PX), dtype=np.float32)
    for t_idx, yr in enumerate(feat_years):
        asset_id = f"{ASSET_FOLDER}/hls_{yr}"
        if yr in hls_assets:
            hls_img = hls_assets[yr]
        else:
            hls_img = ee.Image(asset_id)
            hls_assets[yr] = hls_img

        hls_arr = _compute_chip(ee, hls_img, lon, lat)  # (8, 64, 64)
        cube[t_idx, :N_HLS_BANDS] = hls_arr
        cube[t_idx, N_HLS_BANDS:] = static_arr

    # Loss mask
    yr_codes = [pred_year - 2000 + i for i in range(3)]
    loss_mask = ee.Image(0)
    for code in yr_codes:
        loss_mask = loss_mask.Or(lossyear_img.eq(code))
    loss_mask = loss_mask.toByte().rename("loss")
    mask_arr = _compute_chip(ee, loss_mask, lon, lat)[0]  # (64, 64)

    return np.nan_to_num(cube, nan=0.0), mask_arr.astype(np.uint8)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel GEE requests")
    parser.add_argument("--test", type=int, default=0)
    args = parser.parse_args()

    ee = _init_ee()

    print(f"[1] Loading chip index: {args.index}")
    df = pd.read_parquet(args.index)
    if args.test > 0:
        df = df.head(args.test)
    print(f"    {len(df):,} chips")

    # Pre-build static image (shared across all chips)
    srtm = ee.Image(SRTM_ASSET)
    static_img = (
        srtm.select("elevation").toFloat()
        .addBands(ee.Terrain.slope(srtm).toFloat().rename("slope"))
        .addBands(ee.Image(HANSEN_ASSET).select("treecover2000").toFloat().rename("treecover2000"))
    )
    lossyear_img = ee.Image(HANSEN_ASSET).select("lossyear")

    # Cache HLS asset references
    hls_assets = {}

    progress = _load_progress()
    CHIPS_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name].reset_index(drop=True)
        n = len(split_df)
        if n == 0:
            continue

        out_path = CHIPS_DIR / f"{split_name}_chips.h5"
        start_idx = progress.get(split_name, -1) + 1

        if start_idx >= n:
            print(f"\n[{split_name}] Already complete ({n} chips)")
            continue

        if start_idx > 0:
            print(f"\n[{split_name}] Resuming from chip {start_idx}/{n}")
        else:
            print(f"\n[{split_name}] Extracting {n} chips → {out_path}")
            # Create fresh HDF5
            with h5py.File(out_path, "w") as f:
                f.create_dataset("images",
                    shape=(n, FEATURE_WINDOW, N_CHANNELS, CHIP_PX, CHIP_PX),
                    dtype=np.float32,
                    chunks=(1, FEATURE_WINDOW, N_CHANNELS, CHIP_PX, CHIP_PX),
                    compression="gzip", compression_opts=4)
                f.create_dataset("masks",
                    shape=(n, CHIP_PX, CHIP_PX), dtype=np.uint8,
                    chunks=(1, CHIP_PX, CHIP_PX),
                    compression="gzip", compression_opts=4)
                f.create_dataset("pid", shape=(n,), dtype=np.int64)
                f.create_dataset("pred_year", shape=(n,), dtype=np.int16)
                f.create_dataset("lon", shape=(n,), dtype=np.float64)
                f.create_dataset("lat", shape=(n,), dtype=np.float64)

        t0 = time.time()
        n_ok = 0
        n_fail = 0

        with h5py.File(out_path, "r+") as f:
            # Sequential with throttling (GEE rate limits parallel requests)
            for i in range(start_idx, n):
                row = split_df.iloc[i]
                try:
                    cube, mask = extract_one_chip(
                        ee, float(row.lon), float(row.lat), int(row.pred_year),
                        hls_assets, static_img, lossyear_img,
                    )
                    f["images"][i] = cube
                    f["masks"][i] = mask
                    f["pid"][i] = row.pid
                    f["pred_year"][i] = row.pred_year
                    f["lon"][i] = row.lon
                    f["lat"][i] = row.lat
                    n_ok += 1
                except Exception as e:
                    print(f"    WARN chip {i} (pid={row.pid}): {e}")
                    n_fail += 1

                # Save progress every 10 chips
                if (i + 1) % 10 == 0:
                    progress[split_name] = i
                    _save_progress(progress)
                    f.flush()

                if (i + 1) % 100 == 0 or (i + 1) == n:
                    elapsed = time.time() - t0
                    done = i - start_idx + 1
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (n - i - 1) / rate / 3600 if rate > 0 else 0
                    print(f"    [{split_name}] {i+1}/{n} "
                          f"({n_ok} ok, {n_fail} fail, "
                          f"{rate:.1f} chips/s, ETA {eta:.1f}h)")

                time.sleep(0.1)  # light throttle

        progress[split_name] = n - 1
        _save_progress(progress)
        print(f"    Done: {n_ok} ok, {n_fail} fail → {out_path}")

    print("\nAll splits complete!")


if __name__ == "__main__":
    main()
