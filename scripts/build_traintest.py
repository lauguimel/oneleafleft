"""Build TRAIN features dataset from raw GEE extraction.

Reads the shared raw parquet checkpoint and builds sliding-window features
for training only (pred_yr 2020-2022).

TEMPORAL LEAKAGE GUARANTEE:
- Feature years: 2016-2022 — train NEVER sees 2023 data
- Val (pred_yr=2024, built by build_val.py) uses 2023 as Lag1
- This ensures strict temporal separation: train < 2023 < val

Usage:
    python scripts/build_traintest.py --raw data/raw_250k_YYYYMMDD.parquet
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data.gee_extraction import rebuild_features_dataset

# Feature years: 2016-2022 — train NEVER sees 2023 data
FEATURE_YEARS = list(range(2016, 2023))  # [2016..2022]

# Train predictions only — val is handled by build_val.py
PREDICTION_YEARS = [2020, 2021, 2022]

FEATURE_WINDOW = 4


def main():
    parser = argparse.ArgumentParser(
        description="Build train+test sliding-window dataset (no val)"
    )
    parser.add_argument("--raw", type=str, required=True, help="Path to raw parquet")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_DIR / "data" / "train_test"),
    )
    args = parser.parse_args()

    raw_path = Path(args.raw)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading raw data from {raw_path}...")
    df_raw = pd.read_parquet(raw_path)
    print(f"  {df_raw.shape[0]:,} points × {df_raw.shape[1]} columns")

    print(f"\nBuilding sliding-window features...")
    print(f"  Feature years: {FEATURE_YEARS}")
    print(f"  Prediction years: {PREDICTION_YEARS}")
    print(f"  Feature window: {FEATURE_WINDOW}")

    df = rebuild_features_dataset(
        df_raw,
        years=FEATURE_YEARS,
        prediction_years=PREDICTION_YEARS,
        feature_window=FEATURE_WINDOW,
    )

    # Force all rows to "train"
    df["split"] = "train"

    # Summary
    pred_yrs = sorted(df["prediction_year"].unique())
    pos_rate = df["target"].mean() * 100
    print(f"\nDataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  train: {len(df):>8,} rows — {pos_rate:.2f}% deforested — pred_yr={pred_yrs}")

    # Save
    tag = date.today().strftime("%Y%m%d")
    out_path = out_dir / f"features_traintest_{tag}.parquet"
    df.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved: {out_path} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
