"""PR-AUC improvement: threshold optimization, calibration, advanced metrics.

Phase 4 of the project:
  4.1 — Advanced metrics: F2, Precision@k, lift, Youden J
  4.2 — Spatial post-processing: DBSCAN clustering of positive predictions
  4.3 — Probability calibration: Platt scaling, isotonic, reliability diagram

Usage:
    conda activate deforest
    python scripts/improve_prauc.py
    python scripts/improve_prauc.py --model data/core_model_20260304.json
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    f1_score, fbeta_score, brier_score_loss,
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from train_xgboost import load_dataset, find_latest_parquet, compute_scale_pos_weight
from ablation_study import resolve_group

OUTPUT_DIR = PROJECT_DIR / "data"
PRAUC_DIR = OUTPUT_DIR / "prauc_analysis"
CORE_GROUPS = ["hansen", "spatial", "infra"]


def load_core_model_and_data(model_path, dataset_path):
    """Load or train core model and dataset."""
    import xgboost as xgb
    from train_xgboost import train as xgb_train, optimal_threshold

    if dataset_path is None:
        dataset_path = OUTPUT_DIR / "features_250k_20260228.parquet"
        if not dataset_path.exists():
            dataset_path = find_latest_parquet(OUTPUT_DIR)

    train_df, val_df, test_df, all_feature_cols = load_dataset(dataset_path)
    core_cols = []
    for g in CORE_GROUPS:
        core_cols.extend(resolve_group(g, all_feature_cols))
    core_cols = list(dict.fromkeys(core_cols))
    print(f"  Features: {len(core_cols)}")

    if model_path is not None:
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        print(f"  Model: {model_path.name}")
    else:
        # Train fresh model with current features
        print("  Training fresh core model...")
        X_tr = train_df[core_cols].values
        y_tr = train_df["target"].values.astype(int)
        X_va = val_df[core_cols].values
        y_va = val_df["target"].values.astype(int)
        spw = compute_scale_pos_weight(y_tr)
        model = xgb_train(X_tr, y_tr, X_va, y_va, scale_pos_weight=spw, n_estimators=1000)
        print(f"  Trained ({model.best_iteration} trees)")

    return model, train_df, val_df, test_df, core_cols


def advanced_metrics(y_true, y_proba, pos_rate):
    """Compute comprehensive metrics at various thresholds."""
    # Precision@k
    sorted_idx = np.argsort(-y_proba)
    n = len(y_true)
    n_pos = y_true.sum()

    results = {"n": n, "n_pos": int(n_pos), "pos_rate": float(pos_rate)}

    for pct in [0.5, 1, 2, 5, 10]:
        k = max(1, int(n * pct / 100))
        top_k = sorted_idx[:k]
        prec_at_k = y_true[top_k].mean()
        lift_at_k = prec_at_k / pos_rate if pos_rate > 0 else 0
        recall_at_k = y_true[top_k].sum() / n_pos if n_pos > 0 else 0
        results[f"precision_at_{pct}pct"] = float(prec_at_k)
        results[f"lift_at_{pct}pct"] = float(lift_at_k)
        results[f"recall_at_{pct}pct"] = float(recall_at_k)

    # Optimal thresholds for various objectives
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # F1
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    results["best_f1"] = float(f1_scores[best_f1_idx])
    results["best_f1_threshold"] = float(thresholds[min(best_f1_idx, len(thresholds)-1)])

    # F2 (favors recall)
    f2_scores = 5 * precision * recall / (4 * precision + recall + 1e-10)
    best_f2_idx = np.argmax(f2_scores)
    results["best_f2"] = float(f2_scores[best_f2_idx])
    results["best_f2_threshold"] = float(thresholds[min(best_f2_idx, len(thresholds)-1)])

    # Youden J (TPR - FPR)
    from sklearn.metrics import roc_curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    results["best_youden_j"] = float(j_scores[best_j_idx])
    results["best_youden_threshold"] = float(roc_thresholds[best_j_idx])

    return results


def plot_concentration_curve(y_true, y_proba, output_dir):
    """Plot concentration/lift curve: cumulative recall vs fraction screened."""
    sorted_idx = np.argsort(-y_proba)
    n = len(y_true)
    n_pos = y_true.sum()

    cum_recall = np.cumsum(y_true[sorted_idx]) / n_pos
    frac_screened = np.arange(1, n + 1) / n

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Concentration curve
    ax1.plot(frac_screened * 100, cum_recall * 100, "b-", linewidth=2, label="Model")
    ax1.plot([0, 100], [0, 100], "k--", alpha=0.5, label="Random")
    ax1.set_xlabel("% of area screened")
    ax1.set_ylabel("% of deforestation captured (recall)")
    ax1.set_title("Concentration Curve")
    ax1.legend()
    ax1.set_xlim(0, 20)
    ax1.set_ylim(0, 100)

    # Annotate key points
    for pct in [1, 5, 10]:
        idx = max(0, int(n * pct / 100) - 1)
        recall_pct = cum_recall[idx] * 100
        ax1.annotate(f"{pct}% → {recall_pct:.0f}%",
                     xy=(pct, recall_pct),
                     xytext=(pct + 1, recall_pct - 8),
                     arrowprops=dict(arrowstyle="->", color="red"),
                     fontsize=9, color="red")

    # Lift curve
    lift = cum_recall / (frac_screened + 1e-10)
    ax2.plot(frac_screened * 100, lift, "r-", linewidth=2)
    ax2.set_xlabel("% of area screened")
    ax2.set_ylabel("Lift (vs random)")
    ax2.set_title("Lift Curve")
    ax2.set_xlim(0, 20)
    ax2.axhline(1, color="k", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = output_dir / "concentration_lift_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def spatial_postprocessing(test_df, y_proba, core_cols, threshold, output_dir):
    """DBSCAN spatial post-processing: remove isolated positive predictions."""
    from sklearn.cluster import DBSCAN

    y_pred = (y_proba >= threshold).astype(int)
    pos_mask = y_pred == 1
    n_pos = pos_mask.sum()

    if n_pos < 5:
        print("  Too few positive predictions for DBSCAN")
        return None

    pos_coords = test_df[["lon", "lat"]].values[pos_mask]
    y_true = test_df["target"].values.astype(int)

    results = {"original_n_pos": int(n_pos)}

    for eps_km in [5, 10, 25]:
        eps_deg = eps_km / 111.0  # approximate degrees
        for min_samples in [2, 3, 5]:
            db = DBSCAN(eps=eps_deg, min_samples=min_samples)
            labels = db.fit_predict(pos_coords)

            # Keep only clustered predictions (label != -1)
            clustered_mask = labels != -1
            n_kept = clustered_mask.sum()
            n_removed = n_pos - n_kept

            # Apply filter
            y_pred_filtered = y_pred.copy()
            pos_indices = np.where(pos_mask)[0]
            removed_indices = pos_indices[~clustered_mask]
            y_pred_filtered[removed_indices] = 0

            # Recompute metrics
            pr_auc_filtered = average_precision_score(y_true, y_proba)  # unchanged (proba same)
            f1_filtered = f1_score(y_true, y_pred_filtered, zero_division=0)
            f2_filtered = fbeta_score(y_true, y_pred_filtered, beta=2, zero_division=0)

            # Precision/recall of filtered predictions
            tp = ((y_pred_filtered == 1) & (y_true == 1)).sum()
            fp = ((y_pred_filtered == 1) & (y_true == 0)).sum()
            fn = ((y_pred_filtered == 0) & (y_true == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0

            config = f"eps={eps_km}km, min={min_samples}"
            results[config] = {
                "kept": int(n_kept), "removed": int(n_removed),
                "precision": float(prec), "recall": float(rec),
                "f1": float(f1_filtered), "f2": float(f2_filtered),
            }

            if eps_km == 10 and min_samples == 3:
                print(f"  DBSCAN({config}): kept {n_kept}/{n_pos}, "
                      f"P={prec:.3f}, R={rec:.3f}, F1={f1_filtered:.3f}")

    return results


def calibration_analysis(model, train_df, val_df, test_df, core_cols, output_dir):
    """Probability calibration using Platt scaling and isotonic regression."""
    X_val = val_df[core_cols].values
    y_val = val_df["target"].values.astype(int)
    X_test = test_df[core_cols].values
    y_test = test_df["target"].values.astype(int)

    y_proba_raw = model.predict_proba(X_test)[:, 1]

    # Calibrate using raw probabilities on validation set (manual calibration)
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    y_proba_val = model.predict_proba(X_val)[:, 1]
    methods = {}

    # Platt scaling (sigmoid) — fit logistic regression on raw probas
    platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    platt.fit(y_proba_val.reshape(-1, 1), y_val)
    y_proba_platt = platt.predict_proba(y_proba_raw.reshape(-1, 1))[:, 1]
    methods["Platt (sigmoid)"] = y_proba_platt

    # Isotonic regression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_proba_val, y_val)
    y_proba_isotonic = iso.predict(y_proba_raw)
    methods["Isotonic"] = y_proba_isotonic

    # Reliability diagrams
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    all_probas = {"Raw XGBoost": y_proba_raw, **methods}
    results = {}

    for ax_idx, (name, y_p) in enumerate(all_probas.items()):
        ax = axes[ax_idx]
        prob_true, prob_pred = calibration_curve(y_test, y_p, n_bins=10,
                                                  strategy="quantile")

        ax.plot(prob_pred, prob_true, "bo-", label="Model")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(name)
        ax.legend()

        # Brier score
        brier = brier_score_loss(y_test, y_p)
        pr_auc = average_precision_score(y_test, y_p)
        auc_roc = roc_auc_score(y_test, y_p)
        ax.text(0.05, 0.95, f"Brier: {brier:.4f}\nPR-AUC: {pr_auc:.4f}\nAUC-ROC: {auc_roc:.4f}",
                transform=ax.transAxes, verticalalignment="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        results[name] = {
            "brier_score": float(brier),
            "pr_auc": float(pr_auc),
            "auc_roc": float(auc_roc),
        }

    plt.tight_layout()
    path = output_dir / "reliability_diagrams.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")

    return results


def main(model_path, dataset_path):
    tag = date.today().strftime("%Y%m%d")
    PRAUC_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("PR-AUC IMPROVEMENT ANALYSIS")
    print("=" * 60)

    # Load
    model, train_df, val_df, test_df, core_cols = load_core_model_and_data(
        model_path, dataset_path
    )

    X_test = test_df[core_cols].values
    y_test = test_df["target"].values.astype(int)
    y_proba = model.predict_proba(X_test)[:, 1]
    pos_rate = y_test.mean()

    print(f"\n  Test set: {len(y_test):,} samples, {y_test.sum()} positives "
          f"({pos_rate*100:.2f}%)")
    print(f"  Baseline PR-AUC: {average_precision_score(y_test, y_proba):.4f}")
    print(f"  Baseline AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

    all_results = {}

    # ═══════════════════════════════════════════════════════════════════════
    # 4.1: Advanced metrics
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("4.1 — ADVANCED METRICS")
    print("=" * 60)

    metrics = advanced_metrics(y_test, y_proba, pos_rate)
    all_results["advanced_metrics"] = metrics

    print(f"\n  Precision@k (test set):")
    for pct in [0.5, 1, 2, 5, 10]:
        prec = metrics[f"precision_at_{pct}pct"]
        lift = metrics[f"lift_at_{pct}pct"]
        rec = metrics[f"recall_at_{pct}pct"]
        print(f"    Top {pct:>4}%: P={prec:.3f}  Lift={lift:.0f}×  Recall={rec:.1%}")

    print(f"\n  Optimal thresholds:")
    print(f"    Best F1:  {metrics['best_f1']:.3f} (thresh={metrics['best_f1_threshold']:.3f})")
    print(f"    Best F2:  {metrics['best_f2']:.3f} (thresh={metrics['best_f2_threshold']:.3f})")
    print(f"    Youden J: {metrics['best_youden_j']:.3f} (thresh={metrics['best_youden_threshold']:.3f})")

    # Concentration/lift curves
    print("\n  Plotting concentration & lift curves...")
    plot_concentration_curve(y_test, y_proba, PRAUC_DIR)

    # ═══════════════════════════════════════════════════════════════════════
    # 4.2: Spatial post-processing
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("4.2 — SPATIAL POST-PROCESSING (DBSCAN)")
    print("=" * 60)

    # Use F2-optimal threshold for DBSCAN analysis
    threshold = metrics["best_f2_threshold"]
    print(f"  Using F2-optimal threshold: {threshold:.3f}")

    dbscan_results = spatial_postprocessing(
        test_df, y_proba, core_cols, threshold, PRAUC_DIR
    )
    if dbscan_results:
        all_results["spatial_postprocessing"] = dbscan_results

    # ═══════════════════════════════════════════════════════════════════════
    # 4.3: Probability calibration
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("4.3 — PROBABILITY CALIBRATION")
    print("=" * 60)

    cal_results = calibration_analysis(
        model, train_df, val_df, test_df, core_cols, PRAUC_DIR
    )
    all_results["calibration"] = cal_results

    print("\n  Calibration comparison:")
    for name, res in cal_results.items():
        print(f"    {name:20s}: Brier={res['brier_score']:.4f}  "
              f"PR-AUC={res['pr_auc']:.4f}  AUC-ROC={res['auc_roc']:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════════════════
    results_path = PRAUC_DIR / f"prauc_analysis_{tag}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path.name}")

    print("\n" + "=" * 60)
    print("PR-AUC ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PR-AUC improvement analysis")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    args = parser.parse_args()
    main(args.model, args.dataset)
