"""Train XGBoost on sliding-window deforestation dataset + SHAP analysis.

Usage:
    conda activate deforest
    python scripts/train_xgboost.py                        # auto-detect latest parquet
    python scripts/train_xgboost.py --dataset data/features_250k_20260228.parquet
    python scripts/train_xgboost.py --dataset data/features_5k_20260227.parquet --quick

Output:
    data/model_YYYYMMDD.json        — XGBoost model
    data/results_YYYYMMDD.json      — metrics (AUC, PR-AUC, threshold stats)
    data/shap_importance_YYYYMMDD.csv  — mean |SHAP| per feature
    data/shap_summary_YYYYMMDD.png  — SHAP beeswarm + bar chart
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "data"

# Columns that must NEVER be features
NON_FEATURE_COLS = {
    "pid", "lon", "lat",
    "target", "split", "prediction_year",
    "lossyear",          # raw Hansen lossyear → target leakage
}


def find_latest_parquet(data_dir: Path) -> Path:
    """Return the most recent features_*.parquet in data_dir."""
    candidates = sorted(data_dir.glob("features_*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No features_*.parquet found in {data_dir}")
    latest = candidates[-1]
    print(f"  Auto-detected: {latest.name}")
    return latest


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load parquet, identify feature columns, split into train/val/test."""
    df = pd.read_parquet(path)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Identify feature columns
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"  Feature columns: {len(feature_cols)}")

    # Log NaN summary
    nan_pct = df[feature_cols].isna().mean() * 100
    high_nan = nan_pct[nan_pct > 30]
    if len(high_nan) > 0:
        print(f"  Columns with >30% NaN ({len(high_nan)}): {list(high_nan.index[:5])}{'...' if len(high_nan)>5 else ''}")
    print(f"  Overall NaN rate: {df[feature_cols].isna().mean().mean()*100:.1f}%")
    print(f"  XGBoost handles NaN natively — no imputation needed")

    train = df[df["split"] == "train"]
    val   = df[df["split"] == "val"]
    test  = df[df["split"] == "test"]

    for name, subset in [("train", train), ("val", val), ("test", test)]:
        pos_rate = subset["target"].mean() * 100
        print(f"  {name:5s}: {len(subset):>8,} rows — {pos_rate:.2f}% deforested")

    return train, val, test, feature_cols


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """Scale_pos_weight = (# negatives) / (# positives) for class imbalance."""
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    spw = n_neg / max(n_pos, 1)
    print(f"  scale_pos_weight = {spw:.1f} ({int(n_neg):,} neg / {int(n_pos):,} pos)")
    return spw


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scale_pos_weight: float,
    n_estimators: int = 1000,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: int = 30,
    seed: int = 42,
) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",          # PR-AUC more informative for imbalanced data
        early_stopping_rounds=early_stopping_rounds,
        random_state=seed,
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"  Best iteration: {model.best_iteration} trees")
    return model


def evaluate(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
    threshold: float = 0.5,
) -> dict:
    y_proba = model.predict_proba(X)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    auc_roc = roc_auc_score(y, y_proba)
    auc_pr  = average_precision_score(y, y_proba)

    print(f"\n  [{split_name}] AUC-ROC={auc_roc:.4f}  PR-AUC={auc_pr:.4f}")
    print(classification_report(y, y_pred, target_names=["No loss", "Deforested"],
                                zero_division=0))

    return {
        "split": split_name,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "threshold": threshold,
    }


def optimal_threshold(model, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Find threshold that maximises F1 on validation set."""
    y_proba = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = f1.argmax()
    threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    print(f"  Optimal threshold (max F1 on val): {threshold:.4f} "
          f"→ F1={f1[best_idx]:.4f}, P={precision[best_idx]:.4f}, R={recall[best_idx]:.4f}")
    return threshold


def run_shap(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    feature_cols: list[str],
    output_dir: Path,
    tag: str,
    max_display: int = 20,
) -> pd.Series:
    print("\n  Computing SHAP values...")
    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print(f"  Done in {time.time()-t0:.1f}s")

    importance = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_cols,
    ).sort_values(ascending=False)

    # Save CSV
    csv_path = output_dir / f"shap_importance_{tag}.csv"
    importance.to_csv(csv_path, header=["mean_abs_shap"])
    print(f"  Saved: {csv_path.name}")

    # Plot: bar + beeswarm side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Bar chart
    top = importance.head(max_display)
    axes[0].barh(top.index[::-1], top.values[::-1], color="steelblue")
    axes[0].set_xlabel("Mean |SHAP value|")
    axes[0].set_title(f"Top {max_display} — Global SHAP Importance")
    axes[0].tick_params(axis="y", labelsize=8)

    # SHAP beeswarm (using shap library)
    plt.sca(axes[1])
    top_idx = [feature_cols.index(f) for f in importance.index[:max_display]]
    shap.summary_plot(
        shap_values[:, top_idx],
        X_test[:, top_idx],
        feature_names=[feature_cols[i] for i in top_idx],
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    axes[1].set_title("SHAP Beeswarm")

    plt.tight_layout()
    plot_path = output_dir / f"shap_summary_{tag}.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot_path.name}")

    print(f"\n  Top 15 features by mean |SHAP|:")
    print(importance.head(15).to_string())

    return importance


def main(dataset_path: Path | None, quick: bool) -> None:
    tag = date.today().strftime("%Y%m%d")

    print("=" * 60)
    print("XGBOOST TRAINING — Deforestation Prediction")
    print("=" * 60)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    print("\n[1] Loading dataset...")
    if dataset_path is None:
        dataset_path = find_latest_parquet(OUTPUT_DIR)
    train_df, val_df, test_df, feature_cols = load_dataset(dataset_path)

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values.astype(int)
    X_val   = val_df[feature_cols].values
    y_val   = val_df["target"].values.astype(int)
    X_test  = test_df[feature_cols].values
    y_test  = test_df["target"].values.astype(int)

    # ── 2. Class imbalance ────────────────────────────────────────────────────
    print("\n[2] Class imbalance:")
    spw = compute_scale_pos_weight(y_train)

    # ── 3. Train ──────────────────────────────────────────────────────────────
    print("\n[3] Training XGBoost...")
    t0 = time.time()
    n_est = 200 if quick else 1000
    model = train(X_train, y_train, X_val, y_val,
                  scale_pos_weight=spw, n_estimators=n_est)
    print(f"  Trained in {time.time()-t0:.1f}s")

    # ── 4. Optimal threshold ──────────────────────────────────────────────────
    print("\n[4] Threshold optimisation (on val):")
    threshold = optimal_threshold(model, X_val, y_val)

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    print("\n[5] Evaluation:")
    results = [
        evaluate(model, X_train, y_train, "train", threshold),
        evaluate(model, X_val,   y_val,   "val",   threshold),
        evaluate(model, X_test,  y_test,  "test",  threshold),
    ]

    # ── 6. Save model ─────────────────────────────────────────────────────────
    model_path = OUTPUT_DIR / f"model_{tag}.json"
    model.save_model(str(model_path))
    print(f"\n[6] Model saved: {model_path.name}")

    # Save results JSON
    results_data = {
        "dataset": str(dataset_path),
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "scale_pos_weight": spw,
        "best_iteration": model.best_iteration,
        "threshold": threshold,
        "splits": results,
    }
    results_path = OUTPUT_DIR / f"results_{tag}.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"   Results saved: {results_path.name}")

    # ── 7. SHAP ───────────────────────────────────────────────────────────────
    print("\n[7] SHAP analysis (test set):")
    # Cap SHAP at 10K rows for speed on very large datasets
    n_shap = min(10_000, len(X_test))
    idx = np.random.default_rng(42).choice(len(X_test), n_shap, replace=False)
    importance = run_shap(model, X_test[idx], feature_cols, OUTPUT_DIR, tag)

    # ── 8. ROC curve plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    for split_name, X, y in [
        ("train", X_train, y_train),
        ("val",   X_val,   y_val),
        ("test",  X_test,  y_test),
    ]:
        y_proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc = roc_auc_score(y, y_proba)
        ax.plot(fpr, tpr, label=f"{split_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Deforestation Prediction")
    ax.legend()
    roc_path = OUTPUT_DIR / f"roc_curve_{tag}.png"
    plt.savefig(roc_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"   ROC curve saved: {roc_path.name}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Test AUC-ROC : {results[2]['auc_roc']:.4f}")
    print(f"  Test PR-AUC  : {results[2]['auc_pr']:.4f}")
    print(f"  Top feature  : {importance.index[0]}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost on sliding-window dataset")
    parser.add_argument(
        "--dataset", type=Path, default=None,
        help="Path to features_*.parquet (default: auto-detect latest)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="200 trees instead of 1000 (faster for testing)"
    )
    args = parser.parse_args()
    main(args.dataset, args.quick)
