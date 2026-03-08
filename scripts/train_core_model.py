"""Train XGBoost on core features only (hansen + spatial + infra).

Based on ablation study results (2026-03-03), these 3 groups give the best
performance: Test AUC-ROC 0.931 with 73 features. Adding more groups degrades.

Usage:
    conda activate deforest
    python scripts/train_core_model.py
    python scripts/train_core_model.py --dataset data/features_250k_20260228.parquet
    python scripts/train_core_model.py --quick

Output:
    data/core_model_YYYYMMDD.json        — XGBoost model
    data/core_results_YYYYMMDD.json      — metrics
    data/core_shap_importance_YYYYMMDD.csv
    data/core_shap_summary_YYYYMMDD.png
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from train_xgboost import (
    load_dataset,
    find_latest_parquet,
    compute_scale_pos_weight,
    train,
    evaluate,
    optimal_threshold,
    run_shap,
)
from ablation_study import resolve_group

OUTPUT_DIR = PROJECT_DIR / "data"
CORE_GROUPS = ["hansen", "spatial", "infra"]


def main(dataset_path: Path | None, quick: bool) -> None:
    tag = date.today().strftime("%Y%m%d")

    print("=" * 60)
    print("CORE MODEL — hansen + spatial + infra")
    print("=" * 60)

    # ── 1. Load and filter to core features ────────────────────────────────
    print("\n[1] Loading dataset...")
    if dataset_path is None:
        dataset_path = find_latest_parquet(OUTPUT_DIR)
    train_df, val_df, test_df, all_feature_cols = load_dataset(dataset_path)

    # Select only core group columns
    core_cols = []
    for g in CORE_GROUPS:
        group_cols = resolve_group(g, all_feature_cols)
        print(f"  {g}: {len(group_cols)} features")
        core_cols.extend(group_cols)
    core_cols = list(dict.fromkeys(core_cols))  # deduplicate, preserve order
    print(f"  Total core features: {len(core_cols)}")

    X_train = train_df[core_cols].values
    y_train = train_df["target"].values.astype(int)
    X_val = val_df[core_cols].values
    y_val = val_df["target"].values.astype(int)
    X_test = test_df[core_cols].values
    y_test = test_df["target"].values.astype(int)

    # ── 2. Class imbalance ──────────────────────────────────────────────────
    print("\n[2] Class imbalance:")
    spw = compute_scale_pos_weight(y_train)

    # ── 3. Train ────────────────────────────────────────────────────────────
    print("\n[3] Training XGBoost (core features)...")
    t0 = time.time()
    n_est = 200 if quick else 1000
    model = train(X_train, y_train, X_val, y_val,
                  scale_pos_weight=spw, n_estimators=n_est)
    print(f"  Trained in {time.time()-t0:.1f}s")

    # ── 4. Optimal threshold ────────────────────────────────────────────────
    print("\n[4] Threshold optimisation (on val):")
    threshold = optimal_threshold(model, X_val, y_val)

    # ── 5. Evaluate ─────────────────────────────────────────────────────────
    print("\n[5] Evaluation:")
    results = [
        evaluate(model, X_train, y_train, "train", threshold),
        evaluate(model, X_val,   y_val,   "val",   threshold),
        evaluate(model, X_test,  y_test,  "test",  threshold),
    ]

    # ── 6. Save model ───────────────────────────────────────────────────────
    model_path = OUTPUT_DIR / f"core_model_{tag}.json"
    model.save_model(str(model_path))
    print(f"\n[6] Model saved: {model_path.name}")

    results_data = {
        "dataset": str(dataset_path),
        "core_groups": CORE_GROUPS,
        "feature_cols": core_cols,
        "n_features": len(core_cols),
        "scale_pos_weight": spw,
        "best_iteration": model.best_iteration,
        "threshold": threshold,
        "splits": results,
    }
    results_path = OUTPUT_DIR / f"core_results_{tag}.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"  Results saved: {results_path.name}")

    # ── 7. SHAP ─────────────────────────────────────────────────────────────
    print("\n[7] SHAP analysis (test set):")
    n_shap = min(10_000, len(X_test))
    idx = np.random.default_rng(42).choice(len(X_test), n_shap, replace=False)
    importance = run_shap(model, X_test[idx], core_cols, OUTPUT_DIR,
                          f"core_{tag}")

    print("\n" + "=" * 60)
    print("CORE MODEL COMPLETE")
    print(f"  Features: {len(core_cols)} ({', '.join(CORE_GROUPS)})")
    print(f"  Test AUC-ROC : {results[2]['auc_roc']:.4f}")
    print(f"  Test PR-AUC  : {results[2]['auc_pr']:.4f}")
    print(f"  Top feature  : {importance.index[0]}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train core XGBoost model")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    main(args.dataset, args.quick)
