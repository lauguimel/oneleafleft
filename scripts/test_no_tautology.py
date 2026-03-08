"""Quick test: compare XGBoost with vs without tautological features.

forest_remaining_w and loss_last2yrs_w are derived from cum_deforested_Lag1,
which is 1 for pixels already deforested within the feature window.
These pixels have target=0 by definition (deforestation is irreversible),
so the features provide trivial classification for a large fraction of data.

This script trains two models (with / without) and compares AUC-ROC + PR-AUC
on all splits, plus on the interesting subset (pixels with forest remaining).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

# Reuse functions from the main training script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_xgboost import (
    load_dataset,
    compute_scale_pos_weight,
    train,
    find_latest_parquet,
    NON_FEATURE_COLS,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "data"

TAUTOLOGICAL_FEATURES = {"forest_remaining_w", "loss_last2yrs_w"}


def evaluate_quick(model, X, y, feature_cols, split_name):
    y_proba = model.predict_proba(X)[:, 1]
    auc_roc = roc_auc_score(y, y_proba)
    auc_pr = average_precision_score(y, y_proba)
    return {"split": split_name, "auc_roc": auc_roc, "auc_pr": auc_pr,
            "n": len(y), "n_pos": int(y.sum())}


def main():
    print("=" * 70)
    print("TEST: Impact of tautological features (forest_remaining_w, loss_last2yrs_w)")
    print("=" * 70)

    # Load the 250k dataset specifically
    dataset_path = OUTPUT_DIR / "features_250k_20260228.parquet"
    if not dataset_path.exists():
        dataset_path = find_latest_parquet(OUTPUT_DIR)
    train_df, val_df, test_df, feature_cols_full = load_dataset(dataset_path)

    # Identify tautological columns that are actually present
    tauto_present = [c for c in feature_cols_full if c in TAUTOLOGICAL_FEATURES]
    print(f"\n  Tautological features found: {tauto_present}")

    # Feature sets: with and without
    feature_cols_clean = [c for c in feature_cols_full if c not in TAUTOLOGICAL_FEATURES]
    print(f"  Full features: {len(feature_cols_full)}")
    print(f"  Clean features: {len(feature_cols_clean)}")

    # Count already-deforested pixels in each split
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if "forest_remaining_w" in df.columns:
            n_dead = (df["forest_remaining_w"] <= 0.01).sum()
            pct = n_dead / len(df) * 100
            print(f"  {name}: {n_dead:,} / {len(df):,} pixels with forest_remaining ≈ 0 ({pct:.1f}%)")

    results = {}

    for label, feat_cols in [("WITH tautological", feature_cols_full),
                              ("WITHOUT tautological", feature_cols_clean)]:
        print(f"\n{'='*70}")
        print(f"  Training {label} ({len(feat_cols)} features)")
        print(f"{'='*70}")

        X_train = train_df[feat_cols].values
        y_train = train_df["target"].values.astype(int)
        X_val = val_df[feat_cols].values
        y_val = val_df["target"].values.astype(int)
        X_test = test_df[feat_cols].values
        y_test = test_df["target"].values.astype(int)

        spw = compute_scale_pos_weight(y_train)

        t0 = time.time()
        model = train(X_train, y_train, X_val, y_val,
                      scale_pos_weight=spw, n_estimators=1000)
        elapsed = time.time() - t0
        print(f"  Trained in {elapsed:.1f}s")

        split_results = []
        for sname, X, y in [("train", X_train, y_train),
                             ("val", X_val, y_val),
                             ("test", X_test, y_test)]:
            r = evaluate_quick(model, X, y, feat_cols, sname)
            split_results.append(r)
            print(f"  [{sname:5s}] AUC-ROC={r['auc_roc']:.4f}  PR-AUC={r['auc_pr']:.4f}  "
                  f"(n={r['n']:,}, pos={r['n_pos']:,})")

        # Evaluate on INTERESTING subset: pixels with remaining forest > 0
        if "forest_remaining_w" in test_df.columns:
            mask = test_df["forest_remaining_w"].values > 0.01
            if mask.sum() > 0 and y_test[mask].sum() > 0:
                X_sub = test_df[feat_cols].values[mask]
                y_sub = y_test[mask]
                r = evaluate_quick(model, X_sub, y_sub, feat_cols, "test_forested")
                print(f"  [test_forested] AUC-ROC={r['auc_roc']:.4f}  PR-AUC={r['auc_pr']:.4f}  "
                      f"(n={r['n']:,}, pos={r['n_pos']:,})")
                split_results.append(r)

        results[label] = split_results

    # Summary comparison
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'':30s} {'AUC-ROC':>10s} {'PR-AUC':>10s}")
    print("-" * 52)
    for label, splits in results.items():
        for r in splits:
            print(f"  {label[:7]:7s} {r['split']:15s} {r['auc_roc']:10.4f} {r['auc_pr']:10.4f}")
        print()


if __name__ == "__main__":
    main()
