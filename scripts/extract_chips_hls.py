"""Extract HLS chips + Hansen segmentation masks for Prithvi-EO training.

For each chip in the chip index (from build_chip_dataset.py):
1. Extract T=5 annual median composites from HLS (Harmonized Landsat Sentinel-2)
   at 30 m resolution, cloud-masked, 64×64 pixels.
2. Extract static layers: SRTM (elevation, slope), Hansen treecover2000.
3. Extract the label mask: Hansen lossyear 64×64, binary per pixel
   (1 = loss in [pred_year, pred_year+2]).
4. Save to HDF5 sharded by split (train/val/test).

HLS bands extracted: B02 (blue), B03 (green), B04 (red), B05 (NIR),
  B06 (SWIR1), B07 (SWIR2).  NDVI and NBR computed server-side.

Output: data/chips/{split}_chips.h5
  Datasets:
    - images: (N, T, C, H, W) float32 — T=5 timesteps, C=8 bands + 3 static
    - masks:  (N, H, W) uint8 — binary segmentation target
    - meta:   (N,) compound — pid, pred_year, lon, lat, chip_type, block_id

Usage:
    conda run -n deforest python scripts/extract_chips_hls.py
    conda run -n deforest python scripts/extract_chips_hls.py --index data/chip_index.parquet
    conda run -n deforest python scripts/extract_chips_hls.py --test 100  # smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
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
CHIP_RES = 30  # metres
FEATURE_WINDOW = 5

HANSEN_ASSET = "UMD/hansen/global_forest_change_2024_v1_12"
SRTM_ASSET = "USGS/SRTMGL1_003"
HLS_COLLECTION = "NASA/HLS/HLSL30/v002"  # Landsat component of HLS

# HLS L30 band names: B1(coastal), B2(blue), B3(green), B4(red), B5(NIR), B6(SWIR1), B7(SWIR2)
HLS_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7"]
N_SPECTRAL = len(HLS_BANDS) + 2  # + NDVI + NBR
N_STATIC = 3  # elevation, slope, treecover2000
N_CHANNELS = N_SPECTRAL + N_STATIC  # 11 total

INDEX_PATH = DATA_DIR / "chip_index.parquet"
BATCH_SIZE = 50  # chips per GEE request (computePixels is heavier than reduceRegions)


# ── GEE helpers ───────────────────────────────────────────────────────────────

def _init_gee():
    import ee
    from data.gee_utils import init_gee
    init_gee()
    return ee


def build_annual_composite(ee_module, year: int, aoi: "ee.Geometry") -> "ee.Image":
    """Build a cloud-masked annual median composite from HLS L30 for one year."""
    ee = ee_module

    col = (
        ee.ImageCollection(HLS_COLLECTION)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filterBounds(aoi)
    )

    def mask_clouds(img):
        """Use Fmask band to mask clouds, cloud shadows, and snow."""
        fmask = img.select("Fmask")
        # Bits: 1=cloud, 2=adjacent cloud, 3=cloud shadow, 4=snow
        clear = fmask.bitwiseAnd(0b11110).eq(0)
        return img.updateMask(clear)

    col_masked = col.map(mask_clouds)

    # Median composite of spectral bands
    composite = col_masked.select(HLS_BANDS).median().toFloat()

    # NDVI = (NIR - Red) / (NIR + Red) = (B5 - B4) / (B5 + B4)
    ndvi = composite.normalizedDifference(["B5", "B4"]).rename("NDVI")
    # NBR = (NIR - SWIR2) / (NIR + SWIR2) = (B5 - B7) / (B5 + B7)
    nbr = composite.normalizedDifference(["B5", "B7"]).rename("NBR")

    return composite.addBands(ndvi).addBands(nbr)


def build_static_layers(ee_module) -> "ee.Image":
    """Stack static layers: SRTM elevation + slope + Hansen treecover2000."""
    ee = ee_module
    srtm = ee.Image(SRTM_ASSET)
    elevation = srtm.select("elevation").toFloat().rename("elevation")
    slope = ee.Terrain.slope(srtm).toFloat().rename("slope")
    treecover = (
        ee.Image(HANSEN_ASSET)
        .select("treecover2000")
        .toFloat()
        .rename("treecover2000")
    )
    return elevation.addBands(slope).addBands(treecover)


def build_loss_mask(ee_module, pred_year: int) -> "ee.Image":
    """Binary mask: 1 where Hansen lossyear in [pred_year, pred_year+2], 0 otherwise."""
    ee = ee_module
    lossyear = ee.Image(HANSEN_ASSET).select("lossyear")
    codes = [pred_year - 2000 + i for i in range(3)]
    mask = ee.Image(0)
    for code in codes:
        mask = mask.Or(lossyear.eq(code))
    return mask.rename("loss_mask").toByte()


def extract_chip(
    ee_module,
    lon: float,
    lat: float,
    pred_year: int,
    static_img: "ee.Image",
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract one chip: (T, C, H, W) image array + (H, W) mask array.

    Returns (None, None) on failure.
    """
    ee = ee_module

    pt = ee.Geometry.Point([lon, lat])
    # Square chip region
    chip_region = pt.buffer(CHIP_PX * CHIP_RES / 2).bounds()

    # Build temporal stack
    feat_years = list(range(pred_year - FEATURE_WINDOW, pred_year))
    composites = []
    for yr in feat_years:
        comp = build_annual_composite(ee, yr, chip_region)
        composites.append(comp)

    # Stack all years + static
    loss_mask = build_loss_mask(ee, pred_year)

    # Extract each year as a chip
    try:
        images = []
        for comp in composites:
            # Add static bands to each timestep
            full = comp.addBands(static_img)
            arr = _compute_pixels(ee, full, chip_region)
            if arr is None:
                return None, None
            images.append(arr)

        # Extract mask
        mask_arr = _compute_pixels(ee, loss_mask, chip_region)
        if mask_arr is None:
            return None, None

        # Stack: (T, C, H, W)
        image_cube = np.stack(images, axis=0)
        # Mask: (H, W)
        mask_2d = mask_arr[0]  # single band

        return image_cube, mask_2d

    except Exception as e:
        print(f"    ERROR extracting chip at ({lon:.4f}, {lat:.4f}): {e}")
        return None, None


def _compute_pixels(
    ee_module,
    image: "ee.Image",
    region: "ee.Geometry",
) -> np.ndarray | None:
    """Use ee.data.computePixels to get a numpy array (C, H, W)."""
    ee = ee_module
    try:
        result = ee.data.computePixels({
            "expression": image,
            "fileFormat": "NUMPY_NDARRAY",
            "grid": {
                "dimensions": {"width": CHIP_PX, "height": CHIP_PX},
                "affineTransform": _get_transform(ee, region),
                "crsCode": "EPSG:4326",
            },
        })
        # result is a structured numpy array with band names as fields
        bands = list(result.dtype.names)
        arr = np.stack([result[b].astype(np.float32) for b in bands], axis=0)
        return arr  # (C, H, W)
    except Exception as e:
        print(f"    computePixels error: {e}")
        return None


def _get_transform(ee_module, region: "ee.Geometry") -> dict:
    """Compute affine transform for a chip region at CHIP_RES resolution."""
    # Get bounding box coordinates
    coords = region.bounds().coordinates().getInfo()[0]
    min_lon = min(c[0] for c in coords)
    max_lat = max(c[1] for c in coords)

    # Approximate pixel size in degrees at the equator-ish latitude
    # For 30m: ~0.00027 degrees at equator
    px_deg = CHIP_RES / 111320.0  # approximate

    return {
        "scaleX": px_deg,
        "shearX": 0,
        "translateX": min_lon,
        "shearY": 0,
        "scaleY": -px_deg,
        "translateY": max_lat,
    }


# ── HDF5 writer ───────────────────────────────────────────────────────────────

def create_h5(path: Path, n_chips: int):
    """Create an HDF5 file with preallocated datasets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "images",
            shape=(n_chips, FEATURE_WINDOW, N_CHANNELS, CHIP_PX, CHIP_PX),
            dtype=np.float32,
            chunks=(1, FEATURE_WINDOW, N_CHANNELS, CHIP_PX, CHIP_PX),
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "masks",
            shape=(n_chips, CHIP_PX, CHIP_PX),
            dtype=np.uint8,
            chunks=(1, CHIP_PX, CHIP_PX),
            compression="gzip",
            compression_opts=4,
        )
        # Metadata as separate datasets (simpler than compound)
        f.create_dataset("pid", shape=(n_chips,), dtype=np.int64)
        f.create_dataset("pred_year", shape=(n_chips,), dtype=np.int16)
        f.create_dataset("lon", shape=(n_chips,), dtype=np.float64)
        f.create_dataset("lat", shape=(n_chips,), dtype=np.float64)


# ── Resumable tracker ─────────────────────────────────────────────────────────

PROGRESS_FILE = CHIPS_DIR / "_progress.json"


def _load_progress() -> dict:
    """Load extraction progress: {split: last_completed_index}."""
    import json
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {}


def _save_progress(progress: dict) -> None:
    import json
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress))


# ── Main extraction loop (resumable) ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--test", type=int, default=0,
                        help="Extract only N chips (smoke test)")
    parser.add_argument("--split", type=str, default=None,
                        choices=["train", "val", "test"],
                        help="Extract only one split")
    args = parser.parse_args()

    ee = _init_gee()

    print(f"[1] Loading chip index: {args.index}")
    df = pd.read_parquet(args.index)
    print(f"    {len(df):,} chips total")

    if args.split:
        df = df[df["split"] == args.split].copy()
        print(f"    filtered to split='{args.split}': {len(df):,}")

    if args.test > 0:
        df = df.head(args.test)
        print(f"    smoke test: {len(df)} chips")

    # Build static layers once
    print("[2] Building static layers (SRTM + treecover2000)...")
    static_img = build_static_layers(ee)

    progress = _load_progress()

    # Process by split
    for split_name in df["split"].unique():
        split_df = df[df["split"] == split_name].reset_index(drop=True)
        n = len(split_df)
        out_path = CHIPS_DIR / f"{split_name}_chips.h5"

        # Resume: skip already-completed chips
        start_idx = progress.get(split_name, -1) + 1
        if start_idx > 0:
            print(f"\n[3] Resuming split='{split_name}' from chip {start_idx}/{n}")
        else:
            print(f"\n[3] Extracting {n:,} chips for split='{split_name}' → {out_path}")

        # Create HDF5 only if starting fresh
        if start_idx == 0:
            create_h5(out_path, n)

        n_ok = 0
        n_fail = 0
        t0 = time.time()

        with h5py.File(out_path, "r+") as f:
            for i in range(start_idx, n):
                row = split_df.iloc[i]
                image_cube, mask = extract_chip(
                    ee, row.lon, row.lat, int(row.pred_year), static_img,
                )

                if image_cube is not None and mask is not None:
                    f["images"][i] = image_cube
                    f["masks"][i] = mask
                    f["pid"][i] = row.pid
                    f["pred_year"][i] = row.pred_year
                    f["lon"][i] = row.lon
                    f["lat"][i] = row.lat
                    n_ok += 1
                else:
                    n_fail += 1

                # Save progress every 10 chips
                if (i + 1) % 10 == 0:
                    progress[split_name] = i
                    _save_progress(progress)
                    f.flush()

                if (i + 1) % 50 == 0 or (i + 1) == n:
                    elapsed = time.time() - t0
                    done = i - start_idx + 1
                    rate = done / elapsed if elapsed > 0 else 0
                    remaining = n - i - 1
                    eta = remaining / rate if rate > 0 else 0
                    print(
                        f"    [{split_name}] {i+1}/{n} "
                        f"({n_ok} ok, {n_fail} fail, "
                        f"{rate:.1f} chips/s, ETA {eta/60:.0f}min)"
                    )

                time.sleep(0.3)  # GEE throttle

        # Mark split as fully complete
        progress[split_name] = n - 1
        _save_progress(progress)
        print(f"    Done: {n_ok} ok, {n_fail} failed")
        print(f"    File: {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
