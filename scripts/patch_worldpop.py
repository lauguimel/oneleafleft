"""Re-extract WorldPop for all countries in the study area and patch raw parquet.

Fixes the ~50% NaN from the COD-only filter by using filterBounds (multi-country mosaic).
Loads an existing raw parquet, replaces pop_* columns, saves updated raw + rebuilds features.

Usage:
    conda activate deforest
    python scripts/patch_worldpop.py --raw data/raw_250k_20260228.parquet [--dry-run]
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data.gee_utils import init_gee
from data.gee_extraction import extract_worldpop, build_temporal_features, build_sliding_window_dataset

YEARS = [2016, 2017, 2018, 2019, 2020, 2021]
PREDICTION_YEARS = [2019, 2020, 2021, 2022]


def main(raw_path: Path, dry_run: bool) -> None:
    print("=" * 60)
    print("WORLDPOP PATCH — multi-country mosaic")
    print("=" * 60)

    # ── 1. Load raw checkpoint ────────────────────────────────────────────────
    print(f"\nLoading: {raw_path.name}")
    df_raw = pd.read_parquet(raw_path)
    print(f"  Shape: {df_raw.shape}")

    # Check current coverage
    pop_cols = [c for c in df_raw.columns if c.startswith("pop_")]
    if pop_cols:
        nan_before = df_raw[pop_cols[0]].isna().mean() * 100
        print(f"  WorldPop NaN before: {nan_before:.1f}%")
    else:
        print("  No pop_* columns found")

    if dry_run:
        print("\n[DRY RUN] Would re-extract WorldPop — exiting")
        return

    # ── 2. Init GEE ───────────────────────────────────────────────────────────
    print("\nInitializing GEE...")
    init_gee()
    print("  OK")

    # ── 3. Re-extract WorldPop (multi-country mosaic) ─────────────────────────
    print("\nRe-extracting WorldPop (filterBounds, all countries)...")
    # Reconstruct points DataFrame from raw (lon/lat indexed by pid)
    points = df_raw[["lon", "lat"]].copy()
    points.index.name = "pid"

    t0 = time.time()
    df_pop_new = extract_worldpop(points, years=YEARS)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s — {df_pop_new.shape[1]} columns")

    for yr in YEARS:
        col = f"pop_{yr}"
        if col in df_pop_new.columns:
            nan_pct = df_pop_new[col].isna().mean() * 100
            print(f"    {col}: {nan_pct:.1f}% NaN ({df_pop_new[col].notna().sum():,} valid)")

    # ── 4. Patch raw parquet ──────────────────────────────────────────────────
    print("\nPatching raw parquet...")
    # Drop old pop columns, join new ones
    df_raw_patched = df_raw.drop(columns=[c for c in df_raw.columns if c.startswith("pop_")])
    df_raw_patched = df_raw_patched.join(df_pop_new)

    # Save patched raw (overwrite)
    df_raw_patched.to_parquet(raw_path)
    print(f"  Saved patched raw: {raw_path.name}")

    # ── 5. Rebuild features ───────────────────────────────────────────────────
    print("\nRebuilding features dataset...")
    df_eng = df_raw_patched.copy()

    df_eng = build_temporal_features(df_eng, "pop", YEARS)
    for stat in ["precip_total", "dry_days", "extreme_rain_days"]:
        df_eng = build_temporal_features(df_eng, stat, YEARS)

    if "lossyear" in df_eng.columns:
        for yr in YEARS:
            yr_code = yr - 2000
            df_eng[f"deforested_{yr}"] = (df_eng["lossyear"] == yr_code).astype(float)
        df_eng["cum_loss_before"] = df_eng["lossyear"].between(1, YEARS[0] - 2000 - 1).astype(float)
        df_eng["loss_last2yrs"] = df_eng[[f"deforested_{yr}" for yr in YEARS[-2:]]].sum(axis=1)
        df_eng["forest_remaining"] = (
            df_eng["treecover2000"] / 100.0
            - df_eng[[f"deforested_{yr}" for yr in YEARS]].sum(axis=1) / 100.0
        ).clip(lower=0)

    lossyear_raw = df_raw_patched["lossyear"].fillna(0)
    df_dataset = build_sliding_window_dataset(df_eng, lossyear_raw, PREDICTION_YEARS, feature_window=4)
    print(f"  Dataset shape: {df_dataset.shape}")

    for split in ["train", "val", "test"]:
        sub = df_dataset[df_dataset["split"] == split]
        pos_rate = sub["target"].mean() * 100 if len(sub) > 0 else 0
        print(f"  {split:5s}: {len(sub):>7,} rows — {pos_rate:.1f}% deforested")

    # Derive output path from raw path: raw_250k_YYYYMMDD → features_250k_YYYYMMDD
    stem = raw_path.stem.replace("raw_", "features_")
    out_path = raw_path.parent / f"{stem}.parquet"
    df_dataset.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path.name}")

    print("\n" + "=" * 60)
    print("WORLDPOP PATCH COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch WorldPop to multi-country mosaic")
    parser.add_argument("--raw", type=Path, required=True, help="Path to raw_*.parquet")
    parser.add_argument("--dry-run", action="store_true", help="Check NaN without re-extracting")
    args = parser.parse_args()
    main(args.raw, args.dry_run)
