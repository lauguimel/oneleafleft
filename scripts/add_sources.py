"""Add ERA5, VIIRS, and optionally MODIS fire to an existing raw parquet checkpoint.

This script patches a raw_*.parquet file produced by scale_up_extraction.py by
adding new GEE sources without re-running the full extraction.

Usage:
    conda activate deforest
    python scripts/add_sources.py --raw data/raw_250k_20260228.parquet
    python scripts/add_sources.py --raw data/raw_250k_20260228.parquet --skip-fire
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data.gee_utils import init_gee
from data.gee_extraction import (
    extract_era5,
    extract_viirs,
    extract_modis_fire,
    build_temporal_features,
    build_sliding_window_dataset,
)

YEARS = [2016, 2017, 2018, 2019, 2020, 2021]
PREDICTION_YEARS = [2019, 2020, 2021, 2022]


def main(raw_path: Path, skip_fire: bool) -> None:
    print("=" * 60)
    print("ADD SOURCES — ERA5 + VIIRS" + (" (no fire)" if skip_fire else " + MODIS fire"))
    print("=" * 60)

    # ── 1. Load raw checkpoint ────────────────────────────────────────────────
    print(f"\nLoading: {raw_path.name}")
    df_raw = pd.read_parquet(raw_path)
    n_pts = len(df_raw)
    print(f"  {n_pts:,} points, {df_raw.shape[1]} columns")

    points = df_raw[["lon", "lat"]].copy()
    points.index.name = "pid"

    # ── 2. Init GEE ───────────────────────────────────────────────────────────
    print("\nInitializing GEE...")
    init_gee()
    print("  OK")

    # ── 3. ERA5 ───────────────────────────────────────────────────────────────
    print("\n[1] ERA5-Land annual profiles (temperature, soil moisture, ET)...")
    t0 = time.time()
    df_era5 = extract_era5(points, years=YEARS)
    print(f"  Done in {time.time()-t0:.0f}s — {df_era5.shape[1]} columns")
    for col in [f"temperature_2m_{YEARS[-1]}", f"sm_surface_{YEARS[-1]}"]:
        if col in df_era5.columns:
            print(f"    {col}: {df_era5[col].notna().mean()*100:.0f}% valid")

    # ── 4. VIIRS ──────────────────────────────────────────────────────────────
    print("\n[2] VIIRS nighttime lights annual profiles...")
    t0 = time.time()
    df_viirs = extract_viirs(points, years=YEARS)
    print(f"  Done in {time.time()-t0:.0f}s — {df_viirs.shape[1]} columns")

    # ── 5. MODIS fire (optional) ──────────────────────────────────────────────
    if not skip_fire:
        print("\n[3] MODIS Terra fire annual stats...")
        t0 = time.time()
        df_fire = extract_modis_fire(points, years=YEARS)
        print(f"  Done in {time.time()-t0:.0f}s — {df_fire.shape[1]} columns")
        fire_pct = (df_fire[f"fire_days_{YEARS[-1]}"] > 0).mean() * 100
        print(f"    fire_days_{YEARS[-1]}: {fire_pct:.1f}% of points had fire")
    else:
        df_fire = pd.DataFrame(index=points.index)
        print("\n[3] Skipping MODIS fire (--skip-fire)")

    # ── 6. Patch raw parquet ──────────────────────────────────────────────────
    print("\nPatching raw parquet...")
    # Drop existing columns with the same names (if re-running)
    new_cols = list(df_era5.columns) + list(df_viirs.columns) + list(df_fire.columns)
    df_raw_patched = df_raw.drop(columns=[c for c in new_cols if c in df_raw.columns])
    df_raw_patched = df_raw_patched.join(df_era5).join(df_viirs).join(df_fire)
    df_raw_patched.to_parquet(raw_path)
    print(f"  Saved: {raw_path.name} ({df_raw_patched.shape[1]} columns)")

    # ── 7. Rebuild features ───────────────────────────────────────────────────
    print("\nRebuilding features dataset...")
    df_eng = df_raw_patched.copy()

    df_eng = build_temporal_features(df_eng, "pop", YEARS)
    for stat in ["precip_total", "dry_days", "extreme_rain_days"]:
        df_eng = build_temporal_features(df_eng, stat, YEARS)
    for stat in ["temperature_2m", "sm_surface", "hot_days", "et"]:
        if any(f"{stat}_{yr}" in df_raw_patched.columns for yr in YEARS):
            df_eng = build_temporal_features(df_eng, stat, YEARS)
    for stat in ["ntl_mean", "ntl_max"]:
        if any(f"{stat}_{yr}" in df_raw_patched.columns for yr in YEARS):
            df_eng = build_temporal_features(df_eng, stat, YEARS)
    for stat in ["fire_days"]:
        if any(f"{stat}_{yr}" in df_raw_patched.columns for yr in YEARS):
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

    stem = raw_path.stem.replace("raw_", "features_")
    out_path = raw_path.parent / f"{stem}.parquet"
    df_dataset.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path.name}")

    print("\n" + "=" * 60)
    print("ADD SOURCES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add ERA5/VIIRS/MODIS fire to raw parquet")
    parser.add_argument("--raw", type=Path, required=True, help="Path to raw_*.parquet")
    parser.add_argument("--skip-fire", action="store_true",
                        help="Skip MODIS fire (faster; ~6h for 250K)")
    args = parser.parse_args()
    main(args.raw, args.skip_fire)
