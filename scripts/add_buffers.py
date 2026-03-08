"""Add Hansen spatial buffer features to an existing raw parquet checkpoint.

Downloads buffered deforestation images as tiled GeoTIFFs via computePixels,
then samples at point locations with rasterio. Encodes spatial contagion:
deforestation in the neighbourhood predicts future local deforestation.

Columns added: defo_rate_{r}m_{yr}  (e.g. defo_rate_500m_2019)

Usage:
    conda activate deforest
    python scripts/add_buffers.py --raw data/raw_250k_20260228.parquet
    python scripts/add_buffers.py --raw data/raw_250k_20260228.parquet --buffers 500 5000
    python scripts/add_buffers.py --raw data/raw_250k_20260228.parquet --buffers 150 500 1500 5000

Estimated time: ~30 min for 250K points × 2 radii (raster approach).
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data.gee_utils import init_gee
from data.gee_extraction import extract_hansen_buffers, rebuild_features_dataset

_DEFAULT_YEARS = [2016, 2017, 2018, 2019, 2020, 2021]
_DEFAULT_BUFFERS = [500, 5000]


def main(raw_path: Path, years: list[int], buffers_m: list[int]) -> None:
    prediction_years = list(range(years[0] + 3, years[-1] + 2))

    print("=" * 60)
    print("ADD BUFFERS — Hansen spatial deforestation rates (raster)")
    print(f"  years={years}")
    print(f"  buffers_m={buffers_m}")
    print(f"  method: computePixels + rasterio sampling")
    print("=" * 60)

    # ── 1. Load raw checkpoint ────────────────────────────────────────────────
    print(f"\nLoading: {raw_path.name}")
    df_raw = pd.read_parquet(raw_path)
    print(f"  {len(df_raw):,} points, {df_raw.shape[1]} columns")

    points = df_raw[["lon", "lat"]].copy()
    points.index.name = "pid"

    # ── 2. Init GEE ───────────────────────────────────────────────────────────
    print("\nInitializing GEE...")
    init_gee()
    print("  OK")

    # ── 3. Extract buffers ────────────────────────────────────────────────────
    print(f"\nExtracting Hansen buffers ({len(years)} years × {len(buffers_m)} radii)...")
    t0 = time.time()
    df_buffers = extract_hansen_buffers(points, years=years, buffers_m=buffers_m)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s ({elapsed/60:.1f} min) — {df_buffers.shape[1]} columns")

    # Quick sanity check
    sample_col = f"defo_rate_{buffers_m[-1]}m_{years[-1]}"
    if sample_col in df_buffers.columns:
        pct = (df_buffers[sample_col] > 0).mean() * 100
        mean_val = df_buffers[sample_col].mean()
        print(f"  {sample_col}: {pct:.1f}% points with deforestation, mean={mean_val:.4f}")

    # ── 4. Patch raw parquet ──────────────────────────────────────────────────
    print("\nPatching raw parquet...")
    new_cols = list(df_buffers.columns)
    df_raw_patched = df_raw.drop(columns=[c for c in new_cols if c in df_raw.columns])
    df_raw_patched = df_raw_patched.join(df_buffers)
    df_raw_patched.to_parquet(raw_path)
    print(f"  Saved: {raw_path.name} ({df_raw_patched.shape[1]} columns)")

    # ── 5. Rebuild features dataset ───────────────────────────────────────────
    # Detect all available years from raw columns (not just buffer years)
    import re as _re
    all_years = sorted({
        int(m.group(1))
        for c in df_raw_patched.columns
        for m in [_re.search(r"_(\d{4})$", c)]
        if m and 2000 <= int(m.group(1)) <= 2030
    })
    rebuild_pred_years = list(range(all_years[0] + 3, all_years[-1] + 2))
    print(f"\nRebuilding features dataset...")
    print(f"  detected years={all_years}, prediction_years={rebuild_pred_years}")
    df_dataset = rebuild_features_dataset(df_raw_patched, all_years, rebuild_pred_years)
    print(f"  Dataset shape: {df_dataset.shape}")

    for split in ["train", "val", "test"]:
        sub = df_dataset[df_dataset["split"] == split]
        pos_rate = sub["target"].mean() * 100 if len(sub) > 0 else 0
        print(f"  {split:5s}: {len(sub):>7,} rows — {pos_rate:.1f}% deforested")

    stem = raw_path.stem.replace("raw_", "features_")
    out_path = raw_path.parent / f"{stem}.parquet"
    df_dataset.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path.name}")

    print("\n" + "=" * 60)
    print("ADD BUFFERS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add Hansen spatial buffer features to raw parquet")
    parser.add_argument("--raw", type=Path, required=True, help="Path to raw_*.parquet")
    parser.add_argument("--years", type=int, nargs="+", default=_DEFAULT_YEARS,
                        help="Years to extract buffers for (default: 2016-2021)")
    parser.add_argument("--buffers", type=int, nargs="+", default=_DEFAULT_BUFFERS,
                        dest="buffers_m",
                        help="Buffer radii in metres (default: 150 500 1500 5000)")
    args = parser.parse_args()
    main(args.raw, sorted(args.years), sorted(args.buffers_m))
