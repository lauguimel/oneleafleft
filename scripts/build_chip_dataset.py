"""Build an informative chip dataset for Prithvi-EO segmentation training.

Uses EXISTING buffer features (defo_rate_500m, defo_rate_1500m, treecover2000)
to classify chips — NO GEE scan needed.

Strategy:
1. Load 250K points + spatial blocks + raw features.
2. For each prediction window t, classify chips using local features:
   - front:     defo_rate_500m in [t, t+2] > 1%
   - near:      defo_rate_1500m > 0 but 500m < 1%
   - stable:    defo_rate_1500m == 0, treecover > 50%
   - nonforest: treecover < 30%
3. Pool across 4 windows (t=2019..2022).
4. Sample ~20K chips (30% front, 30% near, 30% stable, 10% nonforest).
5. Save chip index for HLS extraction.

Output: data/chip_index.parquet

Usage:
    conda run -n deforest python scripts/build_chip_dataset.py
    conda run -n deforest python scripts/build_chip_dataset.py --n-total 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

DATA_DIR = PROJECT_DIR / "data"

# ── Config ────────────────────────────────────────────────────────────────────
PREDICTION_YEARS = [2019, 2020, 2021, 2022]
FEATURE_WINDOW = 5

N_TOTAL = 20_000
FRAC_FRONT = 0.30
FRAC_NEAR = 0.30
FRAC_STABLE = 0.30
FRAC_NONFOREST = 0.10

# Thresholds for chip classification
FRONT_RATE_MIN = 0.01     # defo_rate_500m > 1% → active front
NEAR_RATE_MIN = 0.0       # defo_rate_1500m > 0 but 500m < 1%
STABLE_TC_MIN = 50.0      # treecover > 50% for stable forest
NONFOREST_TC_MAX = 30.0   # treecover < 30% for non-forest

RAW_PARQUET = DATA_DIR / "raw_250k_20260228.parquet"
SPATIAL_BLOCKS = DATA_DIR / "spatial_blocks_grid_20260409.parquet"
INDEX_OUTPUT = DATA_DIR / "chip_index.parquet"


# ── Chip classification using existing features ──────────────────────────────

def classify_chips(
    df_raw: pd.DataFrame,
    df_blocks: pd.DataFrame,
    prediction_years: list[int],
) -> pd.DataFrame:
    """Classify (point, window) pairs into chip types using existing buffer features."""
    ly = df_raw["lossyear"].fillna(0)
    tc = df_raw["treecover2000"]

    all_rows = []

    for pred_yr in prediction_years:
        print(f"\n[classify] pred_year={pred_yr}")

        # Exclude already deforested before feature window
        feat_start = pred_yr - FEATURE_WINDOW
        already_lost_max = feat_start - 2000 - 1
        if already_lost_max > 0:
            eligible = ~ly.between(1, already_lost_max)
        else:
            eligible = pd.Series(True, index=df_raw.index)

        # Use the defo_rate at the END of the target window (pred_yr + 2)
        # to capture loss that happened during [pred_yr, pred_yr+2].
        # But we also need to subtract the rate BEFORE the window to isolate
        # loss within [t, t+2]. Use rate at t+2 minus rate at t-1.
        rate_end_yr = min(pred_yr + 2, 2023)  # max available year
        rate_start_yr = pred_yr - 1

        col_500_end = f"defo_rate_500m_{rate_end_yr}"
        col_500_start = f"defo_rate_500m_{rate_start_yr}"
        col_1500_end = f"defo_rate_1500m_{rate_end_yr}"
        col_1500_start = f"defo_rate_1500m_{rate_start_yr}"

        # Windowed deforestation rate (loss in [t, t+2] only)
        if col_500_start in df_raw.columns and col_500_end in df_raw.columns:
            rate_500 = (df_raw[col_500_end] - df_raw[col_500_start]).clip(lower=0)
        elif col_500_end in df_raw.columns:
            rate_500 = df_raw[col_500_end]
        else:
            print(f"  WARNING: missing {col_500_end}, skipping year")
            continue

        if col_1500_start in df_raw.columns and col_1500_end in df_raw.columns:
            rate_1500 = (df_raw[col_1500_end] - df_raw[col_1500_start]).clip(lower=0)
        elif col_1500_end in df_raw.columns:
            rate_1500 = df_raw[col_1500_end]
        else:
            rate_1500 = rate_500  # fallback

        # Classify
        chip_type = pd.Series("other", index=df_raw.index)
        chip_type[tc < NONFOREST_TC_MAX] = "nonforest"
        chip_type[(rate_1500 == 0) & (tc >= STABLE_TC_MIN) & (chip_type == "other")] = "stable"
        chip_type[(rate_1500 > NEAR_RATE_MIN) & (rate_500 <= FRONT_RATE_MIN) & (chip_type == "other")] = "near"
        chip_type[(rate_500 > FRONT_RATE_MIN) & (tc >= NONFOREST_TC_MAX) & (chip_type == "other")] = "front"

        # Apply eligibility
        chip_type[~eligible] = "excluded"

        df_yr = pd.DataFrame({
            "pid": df_raw.index,
            "pred_year": pred_yr,
            "chip_type": chip_type,
            "rate_500": rate_500,
            "rate_1500": rate_1500,
            "treecover": tc,
            "feat_year_start": pred_yr - FEATURE_WINDOW,
            "feat_year_end": pred_yr - 1,
        })
        df_yr = df_yr[df_yr["chip_type"] != "excluded"]

        counts = df_yr["chip_type"].value_counts()
        print(f"  {counts.to_dict()}")
        all_rows.append(df_yr)

    df_all = pd.concat(all_rows, ignore_index=True)

    # Merge spatial blocks + coordinates
    df_all = df_all.merge(
        df_blocks[["pid", "lon", "lat", "block_id", "split"]],
        on="pid", how="inner",
    )
    df_all = df_all[df_all["split"] != "dropped"]

    print(f"\nTotal classified: {len(df_all):,}")
    print(df_all["chip_type"].value_counts().to_string())
    return df_all


# ── Informative sampling ──────────────────────────────────────────────────────

def sample_informative(
    df_classified: pd.DataFrame,
    n_total: int = N_TOTAL,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample chips preserving type proportions and spatial splits."""
    rng = np.random.default_rng(seed)

    targets = {
        "front": int(n_total * FRAC_FRONT),
        "near": int(n_total * FRAC_NEAR),
        "stable": int(n_total * FRAC_STABLE),
        "nonforest": int(n_total * FRAC_NONFOREST),
    }

    split_props = (
        df_classified[df_classified["split"].isin(["train", "val", "test"])]
        ["split"].value_counts(normalize=True)
    )

    sampled = []
    for chip_type, n_target in targets.items():
        pool = df_classified[df_classified["chip_type"] == chip_type]
        if len(pool) == 0:
            print(f"  WARNING: no chips of type '{chip_type}'")
            continue

        for split_name in ["train", "val", "test"]:
            split_pool = pool[pool["split"] == split_name]
            n_split = min(int(round(n_target * split_props.get(split_name, 0))), len(split_pool))
            if n_split == 0:
                continue
            idx = rng.choice(len(split_pool), size=n_split, replace=False)
            sampled.append(split_pool.iloc[idx])
            print(f"  {chip_type}/{split_name}: {n_split} (from {len(split_pool)})")

    df_sampled = pd.concat(sampled, ignore_index=True)

    out_cols = [
        "pid", "lon", "lat", "pred_year", "chip_type",
        "rate_500", "rate_1500", "treecover",
        "block_id", "split",
        "feat_year_start", "feat_year_end",
    ]

    print(f"\n=== Sampled dataset ===")
    print(f"Total: {len(df_sampled):,}")
    print(f"\nBy type:\n{df_sampled['chip_type'].value_counts().to_string()}")
    print(f"\nBy split:\n{df_sampled['split'].value_counts().to_string()}")
    print(f"\nBy type × split:")
    print(df_sampled.groupby(["chip_type", "split"]).size().unstack(fill_value=0))

    return df_sampled[out_cols]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--raw", type=Path, default=RAW_PARQUET)
    parser.add_argument("--blocks", type=Path, default=SPATIAL_BLOCKS)
    parser.add_argument("--n-total", type=int, default=N_TOTAL)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[1] Loading raw features: {args.raw}")
    needed_cols = ["lon", "lat", "lossyear", "treecover2000"]
    # Add all defo_rate columns
    df_peek = pd.read_parquet(args.raw, columns=None)
    rate_cols = [c for c in df_peek.columns if "defo_rate_" in c]
    df_raw = df_peek[needed_cols + rate_cols].copy()
    del df_peek
    print(f"    {len(df_raw):,} points, {len(rate_cols)} buffer rate columns")

    print(f"\n[2] Loading spatial blocks: {args.blocks}")
    df_blocks = pd.read_parquet(args.blocks)
    print(f"    splits: {df_blocks['split'].value_counts().to_dict()}")

    print(f"\n[3] Classifying chips using existing buffer features...")
    df_classified = classify_chips(df_raw, df_blocks, PREDICTION_YEARS)

    print(f"\n[4] Sampling {args.n_total:,} informative chips...")
    df_index = sample_informative(df_classified, n_total=args.n_total, seed=args.seed)

    df_index.to_parquet(INDEX_OUTPUT, index=False)
    print(f"\n[5] Saved: {INDEX_OUTPUT} ({INDEX_OUTPUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
