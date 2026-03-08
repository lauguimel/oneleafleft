"""SHAP deep-dive analysis on core model (hansen + spatial + infra).

Generates:
  - Dependence plots for top features (with interaction coloring)
  - Waterfall plots for individual predictions (TP, FP, FN)
  - Subpopulation analysis (by country, protected area, road proximity)

Usage:
    conda activate deforest
    python scripts/shap_deep_dive.py
    python scripts/shap_deep_dive.py --model data/core_model_20260304.json
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from train_xgboost import load_dataset, find_latest_parquet
from ablation_study import resolve_group

OUTPUT_DIR = PROJECT_DIR / "data"
SHAP_DIR = OUTPUT_DIR / "shap_deep"
CORE_GROUPS = ["hansen", "spatial", "infra"]


def load_core_data(dataset_path: Path | None):
    """Load dataset and filter to core features."""
    if dataset_path is None:
        dataset_path = find_latest_parquet(OUTPUT_DIR)
    train_df, val_df, test_df, all_feature_cols = load_dataset(dataset_path)

    core_cols = []
    for g in CORE_GROUPS:
        core_cols.extend(resolve_group(g, all_feature_cols))
    core_cols = list(dict.fromkeys(core_cols))
    return train_df, val_df, test_df, core_cols


def dependence_plots(shap_values, X_test, feature_cols, importance, output_dir):
    """Generate dependence plots for top features with auto interaction."""
    top_features = importance.head(8).index.tolist()

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    for idx, feat in enumerate(top_features):
        ax = axes[idx // 4, idx % 4]
        feat_idx = feature_cols.index(feat)
        shap.dependence_plot(
            feat_idx, shap_values, X_test,
            feature_names=feature_cols,
            ax=ax, show=False,
        )
        ax.set_title(feat, fontsize=10)

    plt.suptitle("SHAP Dependence Plots — Top 8 Core Features", fontsize=14, y=1.02)
    plt.tight_layout()
    path = output_dir / "shap_dependence_top8.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def waterfall_plots(explainer, shap_values, X_test, y_test, y_proba,
                    feature_cols, output_dir):
    """Waterfall plots for TP, FP, FN examples."""
    # Find examples
    threshold = 0.5  # use lower threshold for finding examples
    y_pred = (y_proba >= threshold).astype(int)

    tp_mask = (y_test == 1) & (y_pred == 1)
    fp_mask = (y_test == 0) & (y_pred == 1)
    fn_mask = (y_test == 1) & (y_pred == 0)

    cases = {}
    if tp_mask.any():
        # Pick TP with highest probability
        tp_idx = np.where(tp_mask)[0]
        best_tp = tp_idx[y_proba[tp_idx].argmax()]
        cases["True Positive"] = best_tp
    if fp_mask.any():
        fp_idx = np.where(fp_mask)[0]
        best_fp = fp_idx[y_proba[fp_idx].argmax()]
        cases["False Positive"] = best_fp
    if fn_mask.any():
        fn_idx = np.where(fn_mask)[0]
        # Pick FN with highest probability (closest to being correct)
        best_fn = fn_idx[y_proba[fn_idx].argmax()]
        cases["False Negative"] = best_fn

    if not cases:
        print("  No examples found for waterfall plots")
        return

    n_cases = len(cases)
    fig, axes = plt.subplots(1, n_cases, figsize=(8 * n_cases, 8))
    if n_cases == 1:
        axes = [axes]

    for ax, (label, idx) in zip(axes, cases.items()):
        plt.sca(ax)
        # Create SHAP Explanation object
        explanation = shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X_test[idx],
            feature_names=feature_cols,
        )
        shap.waterfall_plot(explanation, max_display=15, show=False)
        ax.set_title(f"{label} (p={y_proba[idx]:.3f})", fontsize=11)

    plt.tight_layout()
    path = output_dir / "shap_waterfall_examples.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def subpopulation_analysis(shap_values, X_test_df, feature_cols, importance,
                           output_dir):
    """SHAP importance by subpopulation (country, protected area, road proximity)."""
    top10 = importance.head(10).index.tolist()
    top10_idx = [feature_cols.index(f) for f in top10]

    subpops = {}

    # By protected area status
    if "in_protected" in X_test_df.columns:
        subpops["Inside PA"] = X_test_df["in_protected"] == 1
        subpops["Outside PA"] = X_test_df["in_protected"] == 0

    # By road proximity (median split)
    if "dist_road_km" in X_test_df.columns:
        median_road = X_test_df["dist_road_km"].median()
        subpops[f"Near road (<{median_road:.0f}km)"] = X_test_df["dist_road_km"] < median_road
        subpops[f"Far from road (>{median_road:.0f}km)"] = X_test_df["dist_road_km"] >= median_road

    # By tree cover (high vs low)
    if "treecover2000" in X_test_df.columns:
        subpops["High forest (>60%)"] = X_test_df["treecover2000"] > 60
        subpops["Low forest (<30%)"] = X_test_df["treecover2000"] < 30

    if not subpops:
        print("  No subpopulation columns found")
        return

    # Compute mean |SHAP| for each subpop and top feature
    results = {}
    for label, mask in subpops.items():
        mask_np = mask.values if hasattr(mask, "values") else mask
        if mask_np.sum() == 0:
            continue
        sub_shap = np.abs(shap_values[mask_np][:, top10_idx]).mean(axis=0)
        results[label] = dict(zip(top10, sub_shap))

    if not results:
        print("  All subpopulations empty")
        return

    df_sub = pd.DataFrame(results).T
    print("\n  Mean |SHAP| by subpopulation (top 10 features):")
    print(df_sub.round(3).to_string())

    # Save CSV
    csv_path = output_dir / "shap_subpopulations.csv"
    df_sub.to_csv(csv_path)
    print(f"  Saved: {csv_path.name}")

    # Grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    df_sub.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title("Feature Importance by Subpopulation")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = output_dir / "shap_subpopulations.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def main(model_path: Path | None, dataset_path: Path | None):
    SHAP_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("SHAP DEEP-DIVE — Core Model")
    print("=" * 60)

    # ── Load model ──────────────────────────────────────────────────────────
    if model_path is None:
        candidates = sorted(OUTPUT_DIR.glob("core_model_*.json"))
        if not candidates:
            print("ERROR: No core_model_*.json found. Run train_core_model.py first.")
            sys.exit(1)
        model_path = candidates[-1]
    print(f"\n  Loading model: {model_path.name}")
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    # ── Load data ───────────────────────────────────────────────────────────
    print("\n  Loading dataset...")
    train_df, val_df, test_df, core_cols = load_core_data(dataset_path)

    # Sample test set for SHAP (cap at 10K for speed)
    rng = np.random.default_rng(42)
    n_shap = min(10_000, len(test_df))
    idx = rng.choice(len(test_df), n_shap, replace=False)

    X_test = test_df[core_cols].values[idx]
    y_test = test_df["target"].values[idx].astype(int)
    X_test_df = test_df.iloc[idx][core_cols].reset_index(drop=True)

    # ── Compute SHAP ────────────────────────────────────────────────────────
    print(f"\n  Computing SHAP values on {n_shap:,} test samples...")
    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print(f"  Done in {time.time()-t0:.1f}s")

    y_proba = model.predict_proba(X_test)[:, 1]

    importance = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=core_cols,
    ).sort_values(ascending=False)

    # ── 1. Dependence plots ─────────────────────────────────────────────────
    print("\n[1] Dependence plots...")
    dependence_plots(shap_values, X_test, core_cols, importance, SHAP_DIR)

    # ── 2. Waterfall plots ──────────────────────────────────────────────────
    print("\n[2] Waterfall plots (TP, FP, FN)...")
    waterfall_plots(explainer, shap_values, X_test, y_test, y_proba,
                    core_cols, SHAP_DIR)

    # ── 3. Subpopulation analysis ───────────────────────────────────────────
    print("\n[3] Subpopulation analysis...")
    subpopulation_analysis(shap_values, X_test_df, core_cols, importance, SHAP_DIR)

    # ── 4. Spatial scale comparison: 500m vs 5000m ──────────────────────────
    print("\n[4] Spatial scale comparison (500m vs 5000m)...")
    cols_500m = [c for c in core_cols if "500m" in c]
    cols_5000m = [c for c in core_cols if "5000m" in c]
    shap_500m = sum(np.abs(shap_values[:, core_cols.index(c)]).mean() for c in cols_500m)
    shap_5000m = sum(np.abs(shap_values[:, core_cols.index(c)]).mean() for c in cols_5000m)

    print(f"  Total |SHAP| 500m features ({len(cols_500m)} cols): {shap_500m:.3f}")
    print(f"  Total |SHAP| 5000m features ({len(cols_5000m)} cols): {shap_5000m:.3f}")
    print(f"  Ratio 500m/5000m: {shap_500m/shap_5000m:.2f}")

    # Bar chart comparing buffer scales
    fig, ax = plt.subplots(figsize=(8, 5))
    buffer_data = {"500m": shap_500m, "5000m": shap_5000m}
    ax.bar(buffer_data.keys(), buffer_data.values(), color=["#2196F3", "#FF9800"])
    ax.set_ylabel("Total mean |SHAP value|")
    ax.set_title("Spatial Contagion: 500m vs 5000m Buffer Importance")
    for i, (k, v) in enumerate(buffer_data.items()):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=12)
    path = SHAP_DIR / "shap_buffer_scales.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")

    print("\n" + "=" * 60)
    print("SHAP DEEP-DIVE COMPLETE")
    print(f"  Outputs in: {SHAP_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP deep-dive analysis")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    args = parser.parse_args()
    main(args.model, args.dataset)
