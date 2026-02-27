"""Scale-up extraction: 250K points × 4 GEE sources → Parquet.

Implements:
- Batched GEE extraction (4K pts/call → 63 batches for 250K)
- Sliding window dataset (N examples per location)
- Multi-buffer spatial features (150m, 500m, 1.5km, 5km)
- Temporal feature engineering (Δ1yr, Δ3yr, anomaly)
- Strict temporal split: train ≤2020, val 2021, test 2022

Usage:
    conda activate deforest
    python scripts/scale_up_extraction.py [--n-points N] [--skip-buffers]

Output: data/features_Nk_YYYYMMDD.parquet
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

import ee
from data.gee_utils import init_gee
from data.gee_extraction import (
    extract_srtm,
    extract_hansen_static,
    extract_worldpop,
    extract_chirps,
    extract_hansen_buffers,
    build_temporal_features,
    build_sliding_window_dataset,
)

LEGACY_CSV = Path(
    "/Users/guillaume/Documents/Clouds/UGA/Recherche/deforestation"
    "/src/data/tiles_250000_10N_020E_20231023.csv"
)
OUTPUT_DIR = PROJECT_DIR / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

YEARS = [2016, 2017, 2018, 2019, 2020, 2021]  # feature years
PREDICTION_YEARS = [2019, 2020, 2021, 2022]    # target years (sliding window)
BUFFERS_M = [150, 500, 1500, 5000]             # spatial context scales
SEED = 42


def main(n_points: int, skip_buffers: bool) -> None:
    print("=" * 60)
    print(f"SCALE-UP EXTRACTION — {n_points:,} points")
    print("=" * 60)

    # ── 1. Init GEE ──────────────────────────────────────────────────────────
    print("\nInitializing GEE...")
    init_gee()
    print("  OK")

    # ── 2. Sample points ──────────────────────────────────────────────────────
    print(f"\nLoading legacy CSV...")
    tiles = pd.read_csv(
        LEGACY_CSV,
        usecols=["longitude", "latitude", "lossyear_22_mean"],
    ).rename(columns={"longitude": "lon", "latitude": "lat"})
    tiles["deforested_2022"] = (tiles["lossyear_22_mean"] > 0).astype(int)

    if n_points < len(tiles):
        # Stratified sample
        pos = tiles[tiles["deforested_2022"] == 1].sample(n_points // 2, random_state=SEED)
        neg = tiles[tiles["deforested_2022"] == 0].sample(n_points // 2, random_state=SEED)
        points = pd.concat([pos, neg]).sample(frac=1, random_state=SEED)
    else:
        points = tiles.copy()

    points = points.reset_index(drop=True)
    points.index.name = "pid"
    print(f"  {len(points):,} points — {points['deforested_2022'].mean()*100:.1f}% deforested")

    t_total = time.time()

    # ── 3. Static extraction ──────────────────────────────────────────────────
    print("\n[1/4] SRTM + Hansen static...")
    t0 = time.time()
    df_srtm = extract_srtm(points)
    df_hansen = extract_hansen_static(points)
    df_static = df_srtm.join(df_hansen, how="outer")
    print(f"  Done in {time.time()-t0:.0f}s — {df_static.shape[1]} columns")

    # ── 4. WorldPop ───────────────────────────────────────────────────────────
    print("\n[2/4] WorldPop annual (2016-2020)...")
    t0 = time.time()
    df_pop = extract_worldpop(points, years=YEARS)
    print(f"  Done in {time.time()-t0:.0f}s — {df_pop.shape[1]} columns")
    # Valid coverage stats
    for yr in YEARS:
        col = f"pop_{yr}"
        if col in df_pop.columns:
            pct = df_pop[col].notna().mean() * 100
            print(f"    {col}: {pct:.0f}% valid")

    # ── 5. CHIRPS ─────────────────────────────────────────────────────────────
    print("\n[3/4] CHIRPS annual profiles (2016-2021)...")
    t0 = time.time()
    df_chirps = extract_chirps(points, years=YEARS)
    print(f"  Done in {time.time()-t0:.0f}s — {df_chirps.shape[1]} columns")

    # ── 6. Multi-buffer spatial context ───────────────────────────────────────
    if not skip_buffers:
        print(f"\n[4/4] Hansen deforestation rate at {BUFFERS_M} m buffers...")
        t0 = time.time()
        # Only compute buffers for prediction years (not all years)
        df_buffers = extract_hansen_buffers(
            points, years=PREDICTION_YEARS[:-1], buffers_m=BUFFERS_M
        )
        print(f"  Done in {time.time()-t0:.0f}s — {df_buffers.shape[1]} columns")
    else:
        df_buffers = pd.DataFrame(index=points.index)
        print("\n[4/4] Skipping spatial buffers (--skip-buffers)")

    # ── 7. Merge raw features ─────────────────────────────────────────────────
    print("\nMerging raw features...")
    df_raw = (
        points[["lon", "lat"]]
        .join(df_static)
        .join(df_pop)
        .join(df_chirps)
        .join(df_buffers)
    )
    print(f"  Raw shape: {df_raw.shape}")
    print(f"  NaN: {df_raw.isna().sum().sum():,} cells")

    # Save raw (checkpoint — avoids re-running GEE if feature engineering fails)
    raw_path = OUTPUT_DIR / f"raw_{len(points)//1000}k_{date.today():%Y%m%d}.parquet"
    df_raw.to_parquet(raw_path)
    print(f"  Raw saved to {raw_path.name}")

    # ── 8. Temporal feature engineering ───────────────────────────────────────
    print("\nTemporal feature engineering...")
    df_eng = df_raw.copy()

    # Population temporal profiles
    df_eng = build_temporal_features(df_eng, "pop", YEARS)

    # CHIRPS temporal profiles
    for stat in ["precip_total", "dry_days", "extreme_rain_days"]:
        df_eng = build_temporal_features(df_eng, stat, YEARS)

    # Loss momentum from Hansen lossyear_raw
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

    print(f"  Engineered shape: {df_eng.shape}")

    # ── 9. Build sliding window dataset ───────────────────────────────────────
    print("\nBuilding sliding window dataset...")
    lossyear_raw = df_raw["lossyear"].fillna(0)

    df_dataset = build_sliding_window_dataset(
        df_features=df_eng,
        lossyear_raw=lossyear_raw,
        prediction_years=PREDICTION_YEARS,
        feature_window=4,
    )
    print(f"  Dataset shape: {df_dataset.shape}")

    # Split stats
    for split in ["train", "val", "test"]:
        sub = df_dataset[df_dataset["split"] == split]
        pos_rate = sub["target"].mean() * 100 if len(sub) > 0 else 0
        print(f"  {split:5s}: {len(sub):>7,} rows — {pos_rate:.1f}% deforested")

    # Save final dataset
    out_path = OUTPUT_DIR / f"features_{len(points)//1000}k_{date.today():%Y%m%d}.parquet"
    df_dataset.to_parquet(out_path, index=False)
    print(f"\nDataset saved to {out_path.name}")

    elapsed = time.time() - t_total
    print(f"\nTotal time: {elapsed/60:.1f} min")
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale-up GEE extraction")
    parser.add_argument("--n-points", type=int, default=5000,
                        help="Number of sample points (default: 5000, use 250000 for full)")
    parser.add_argument("--skip-buffers", action="store_true",
                        help="Skip multi-buffer spatial extraction (faster)")
    args = parser.parse_args()
    main(args.n_points, args.skip_buffers)
