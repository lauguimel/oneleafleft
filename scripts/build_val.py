"""Build VALIDATION features dataset from raw GEE extraction.

Reads the shared raw parquet checkpoint and builds sliding-window features
for validation (pred_yr 2024) only.

TEMPORAL LEAKAGE GUARANTEE:
- Feature years: 2017-2023 — val uses 2023 as Lag1 to predict 2024
- Train (built by build_traintest.py) only sees 2016-2022
- Strict temporal separation: train features < 2023 <= val features

Usage:
    python scripts/build_val.py --raw data/raw_250k_YYYYMMDD.parquet
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data.gee_extraction import rebuild_features_dataset

# Feature years: 2017-2023 — includes 2023 as Lag1 for predicting 2024
FEATURE_YEARS = list(range(2017, 2024))  # [2017..2023]

# Predict 2024 — target from Hansen lossyear=24
PREDICTION_YEARS = [2024]

FEATURE_WINDOW = 4


def main():
    parser = argparse.ArgumentParser(
        description="Build validation sliding-window dataset (no train/test)"
    )
    parser.add_argument("--raw", type=str, required=True, help="Path to raw parquet")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_DIR / "data" / "val"),
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

    # Force split to "val" (rebuild_features_dataset assigns "test" to the
    # last prediction year, but here it's val)
    df["split"] = "val"

    # Safety check
    assert list(df["prediction_year"].unique()) == [2024], (
        f"Expected only pred_yr=2024, got {df['prediction_year'].unique()}"
    )

    # Drop target-tautology metadata: `lossyear` equals (prediction_year - 2000)
    # for every positive sample by construction. Inflates PR-AUC ×9 if kept.
    df = df.drop(columns=[c for c in ["lossyear"] if c in df.columns])

    # Summary
    pos_rate = df["target"].mean() * 100
    print(f"\nDataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  val: {len(df):>8,} rows — {pos_rate:.2f}% deforested — pred_yr=[2024]")

    # Save
    tag = date.today().strftime("%Y%m%d")
    out_path = out_dir / f"features_val_{tag}.parquet"
    df.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved: {out_path} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
