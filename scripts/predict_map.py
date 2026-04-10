"""Generate a deforestation probability map via sliding-window inference.

Reads full-extent GeoTIFFs (same as chip_from_tiles.py), runs the trained
Prithvi segmentation model in a sliding window with 50% overlap, and blends
predictions using a Gaussian window to produce a seamless probability map.

Output: Cloud-Optimized GeoTIFF (COG) at 30 m resolution.

Algorithm:
  1. Load model checkpoint
  2. Slide a 64×64 window with stride 32 (50% overlap) over the study area
  3. For each window: read HLS stack + static → model → probability mask
  4. Accumulate: proba_sum += pred * gaussian_window
                 weight_sum += gaussian_window
  5. Final map = proba_sum / weight_sum
  6. Write as COG

Usage:
    conda run -n deforest python scripts/predict_map.py \
        --ckpt models/checkpoints/best.ckpt \
        --pred-year 2024 \
        --out results/forecast_2024.tif

    # On Aqua with GPU:
    python scripts/predict_map.py --ckpt best.ckpt --pred-year 2024
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import Window

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

DATA_DIR = PROJECT_DIR / "data"
TILES_DIR = DATA_DIR / "tiles"

# ── Config ────────────────────────────────────────────────────────────────────
CHIP_PX = 64
STRIDE = 32          # 50% overlap
CHIP_RES = 30        # metres
FEATURE_WINDOW = 5

HLS_BANDS_COUNT = 8  # B2-B7 + NDVI + NBR
STATIC_BANDS = 3     # elevation, slope, treecover2000
N_CHANNELS = HLS_BANDS_COUNT + STATIC_BANDS

STUDY_AREA = {
    "lon_min": 20.0, "lon_max": 30.0,
    "lat_min": 0.0, "lat_max": 10.0,
}

# Minimum treecover to run inference (skip non-forest)
MIN_TREECOVER = 30


# ── Gaussian blending window ─────────────────────────────────────────────────

def gaussian_window_2d(size: int, sigma: float | None = None) -> np.ndarray:
    """2D Gaussian window for smooth blending of overlapping predictions."""
    if sigma is None:
        sigma = size / 4
    x = np.arange(size) - (size - 1) / 2
    g1d = np.exp(-x ** 2 / (2 * sigma ** 2))
    window = np.outer(g1d, g1d).astype(np.float32)
    # Normalize so max = 1
    window /= window.max()
    return window


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: str = "cuda"):
    """Load trained Prithvi segmentation model from checkpoint."""
    import torch

    # TerraTorch saves via Lightning — load the task
    from terratorch.tasks import SemanticSegmentationTask
    task = SemanticSegmentationTask.load_from_checkpoint(str(ckpt_path))
    task.eval()
    task.to(device)
    return task, device


# ── Tile reading ──────────────────────────────────────────────────────────────

def open_tiles(tiles_dir: Path, pred_year: int):
    """Open all GeoTIFFs needed for inference at pred_year.

    Returns: dict of rasterio datasets + reference transform.
    """
    feat_years = list(range(pred_year - FEATURE_WINDOW, pred_year))

    hls_srcs = {}
    for yr in feat_years:
        path = tiles_dir / f"hls_{yr}.tif"
        if path.exists():
            hls_srcs[yr] = rasterio.open(path)
        else:
            print(f"  WARNING: {path} not found")
            hls_srcs[yr] = None

    srtm_src = rasterio.open(tiles_dir / "srtm.tif") if (tiles_dir / "srtm.tif").exists() else None
    tc_src = rasterio.open(tiles_dir / "hansen_treecover2000.tif") if (tiles_dir / "hansen_treecover2000.tif").exists() else None

    # Reference for transform
    ref_src = next((s for s in hls_srcs.values() if s is not None), tc_src)
    return hls_srcs, srtm_src, tc_src, ref_src


def read_chip_from_tiles(
    hls_srcs: dict,
    srtm_src,
    tc_src,
    row: int,
    col: int,
    feat_years: list[int],
) -> np.ndarray:
    """Read a (T, C, 64, 64) chip from open rasterio datasets."""
    cube = np.zeros((FEATURE_WINDOW, N_CHANNELS, CHIP_PX, CHIP_PX), dtype=np.float32)
    win = Window(col, row, CHIP_PX, CHIP_PX)

    # Static layers
    if srtm_src:
        static_srtm = srtm_src.read(window=win, boundless=True, fill_value=0).astype(np.float32)
    else:
        static_srtm = np.zeros((2, CHIP_PX, CHIP_PX), dtype=np.float32)

    if tc_src:
        static_tc = tc_src.read(window=win, boundless=True, fill_value=0).astype(np.float32)
    else:
        static_tc = np.zeros((1, CHIP_PX, CHIP_PX), dtype=np.float32)

    static = np.concatenate([static_srtm, static_tc], axis=0)  # (3, 64, 64)

    for t_idx, yr in enumerate(feat_years[:FEATURE_WINDOW]):
        src = hls_srcs.get(yr)
        if src is not None:
            hls = src.read(window=win, boundless=True, fill_value=0).astype(np.float32)
        else:
            hls = np.zeros((HLS_BANDS_COUNT, CHIP_PX, CHIP_PX), dtype=np.float32)

        cube[t_idx, :HLS_BANDS_COUNT] = hls
        cube[t_idx, HLS_BANDS_COUNT:] = static

    return np.nan_to_num(cube, nan=0.0)


# ── Main inference loop ──────────────────────────────────────────────────────

def predict_map(
    ckpt_path: Path,
    pred_year: int,
    tiles_dir: Path,
    out_path: Path,
    device: str = "cuda",
    batch_size: int = 32,
):
    """Run sliding-window inference and write COG."""
    import torch

    print(f"[1] Loading model: {ckpt_path}")
    model, device = load_model(ckpt_path, device)

    print(f"[2] Opening tiles for pred_year={pred_year}")
    feat_years = list(range(pred_year - FEATURE_WINDOW, pred_year))
    print(f"    Feature years: {feat_years}")
    hls_srcs, srtm_src, tc_src, ref_src = open_tiles(tiles_dir, pred_year)

    if ref_src is None:
        print("ERROR: No reference GeoTIFF found!")
        return

    height = ref_src.height
    width = ref_src.width
    transform = ref_src.transform
    crs = ref_src.crs

    print(f"    Raster: {width}×{height} pixels")
    print(f"    Stride: {STRIDE} px (overlap {100*(1-STRIDE/CHIP_PX):.0f}%)")

    # Pre-compute Gaussian window
    gauss = gaussian_window_2d(CHIP_PX)

    # Accumulation arrays
    proba_sum = np.zeros((height, width), dtype=np.float64)
    weight_sum = np.zeros((height, width), dtype=np.float64)

    # Optional: treecover mask to skip non-forest areas
    if tc_src:
        print("    Loading treecover mask...")
        tc_full = tc_src.read(1).astype(np.float32)
    else:
        tc_full = np.full((height, width), 100, dtype=np.float32)

    # Generate all window positions
    positions = []
    for r in range(0, height - CHIP_PX + 1, STRIDE):
        for c in range(0, width - CHIP_PX + 1, STRIDE):
            # Skip non-forest chips (treecover < threshold over the chip)
            tc_chip = tc_full[r:r + CHIP_PX, c:c + CHIP_PX]
            if tc_chip.mean() < MIN_TREECOVER:
                continue
            positions.append((r, c))

    n_chips = len(positions)
    print(f"    Chips to process: {n_chips:,} (skipped non-forest)")

    # Batched inference
    n_done = 0
    for batch_start in range(0, n_chips, batch_size):
        batch_pos = positions[batch_start:batch_start + batch_size]
        batch_cubes = []

        for r, c in batch_pos:
            cube = read_chip_from_tiles(hls_srcs, srtm_src, tc_src, r, c, feat_years)
            # Prithvi expects (C=6, T, H, W) — extract HLS bands and transpose
            hls_cube = cube[:, :6, :, :]  # (T, 6, H, W)
            hls_cube = hls_cube.transpose(1, 0, 2, 3)  # (6, T, H, W)
            batch_cubes.append(hls_cube)

        batch_tensor = torch.from_numpy(np.stack(batch_cubes)).to(device)

        with torch.no_grad():
            logits = model(batch_tensor)  # (B, 2, H, W)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # (B, H, W)

        # Accumulate with Gaussian weighting
        for idx, (r, c) in enumerate(batch_pos):
            r_end = min(r + CHIP_PX, height)
            c_end = min(c + CHIP_PX, width)
            h = r_end - r
            w = c_end - c

            proba_sum[r:r_end, c:c_end] += probs[idx, :h, :w] * gauss[:h, :w]
            weight_sum[r:r_end, c:c_end] += gauss[:h, :w]

        n_done += len(batch_pos)
        if n_done % 5000 == 0 or n_done == n_chips:
            print(f"    {n_done:,}/{n_chips:,} chips processed")

    # Final probability map
    print("[3] Computing final probability map...")
    # Avoid division by zero
    weight_sum = np.where(weight_sum > 0, weight_sum, 1.0)
    proba_map = (proba_sum / weight_sum).astype(np.float32)

    # Mask non-forest areas
    proba_map[tc_full < MIN_TREECOVER] = 0.0

    # Write COG
    print(f"[4] Writing COG: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": width,
        "height": height,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(proba_map, 1)
        dst.update_tags(
            prediction_year=str(pred_year),
            feature_years=str(feat_years),
            model_checkpoint=str(ckpt_path.name),
            description="Deforestation probability [0,1] — Prithvi-EO segmentation",
        )

    print(f"    Size: {out_path.stat().st_size / 1e6:.1f} MB")
    print(f"    Range: [{proba_map.min():.4f}, {proba_map.max():.4f}]")
    print(f"    Mean (forest pixels): {proba_map[tc_full >= MIN_TREECOVER].mean():.4f}")

    # Cleanup
    for src in hls_srcs.values():
        if src is not None:
            src.close()
    for src in [srtm_src, tc_src]:
        if src is not None:
            src.close()

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, required=True, help="Model checkpoint")
    parser.add_argument("--pred-year", type=int, required=True, help="Year to predict")
    parser.add_argument("--tiles-dir", type=Path, default=TILES_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.out is None:
        args.out = PROJECT_DIR / "results" / f"forecast_{args.pred_year}.tif"

    predict_map(
        ckpt_path=args.ckpt,
        pred_year=args.pred_year,
        tiles_dir=args.tiles_dir,
        out_path=args.out,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
