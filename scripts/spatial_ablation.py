"""Spatial ablation study: buffer radius combinations + subsampling.

Phase 3 of the triple ablation:
  3A — Buffer radius ablation: 500m only, 5000m only, both
  3B — Sampling density: 250K, 100K, 50K points
  3C — Spatial decay analysis: defo_rate vs radius

Usage:
    conda activate deforest
    python scripts/spatial_ablation.py
    python scripts/spatial_ablation.py --quick
"""

import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
sys.path.insert(0, str(PROJECT_DIR / "src" / "data"))

from train_xgboost import (
    add_window_summaries,
    compute_scale_pos_weight,
    train,
    evaluate,
    optimal_threshold,
    NON_FEATURE_COLS,
    _GLOBAL_ANOM_RE,
    _GLOBAL_SUMMARY_SUFFIXES,
)
from ablation_study import resolve_group, FEATURE_GROUPS, _sw, _exact
from gee_extraction import rebuild_features_dataset

OUTPUT_DIR = PROJECT_DIR / "data"
# Airtight split
TRAIN_FEATURE_YEARS = list(range(2016, 2023))
TRAIN_PREDICTION_YEARS = [2020, 2021, 2022]
VAL_FEATURE_YEARS = list(range(2017, 2024))
VAL_PREDICTION_YEARS = [2024]


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Same feature prep as train_xgboost.load_dataset."""
    leaky_anom = [c for c in df.columns if _GLOBAL_ANOM_RE.search(c) and "proxy" not in c]
    if leaky_anom:
        df = df.drop(columns=leaky_anom)
    df = add_window_summaries(df)
    global_summaries = {
        c for c in df.columns
        if any(c.endswith(sfx) for sfx in _GLOBAL_SUMMARY_SUFFIXES)
        and not c.endswith("_wmean") and not c.endswith("_wtrend")
    }
    excluded = NON_FEATURE_COLS | global_summaries
    feature_cols = [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    nan_pct = df[feature_cols].isna().mean() * 100
    structural_nan = [c for c in feature_cols if nan_pct[c] > 45]
    feature_cols = [c for c in feature_cols if c not in structural_nan]
    return df, feature_cols


def core_cols(feat_cols):
    """Keep only hansen + spatial (defo_rate_*) + infra features."""
    h = resolve_group("hansen", feat_cols)
    s = [c for c in feat_cols if c.startswith("defo_rate_")]
    i = resolve_group("infra", feat_cols)
    return list(dict.fromkeys(h + s + i))


def train_and_eval(train_df, val_df, feature_cols, n_est=1000):
    """Train XGBoost and return metrics."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    common = [c for c in feature_cols if c in train_df.columns and c in val_df.columns]
    X_tr, y_tr = train_df[common].values, train_df["target"].values.astype(int)
    X_va, y_va = val_df[common].values, val_df["target"].values.astype(int)

    if y_tr.sum() == 0 or y_va.sum() == 0:
        return {"val_auc_roc": np.nan, "val_pr_auc": np.nan, "n_features": len(common)}

    spw = compute_scale_pos_weight(y_tr)
    model = train(X_tr, y_tr, X_va, y_va, scale_pos_weight=spw, n_estimators=n_est)
    threshold = optimal_threshold(model, X_va, y_va)

    results = {}
    for name, X, y in [("train", X_tr, y_tr), ("val", X_va, y_va)]:
        y_proba = model.predict_proba(X)[:, 1]
        results[f"{name}_auc_roc"] = float(roc_auc_score(y, y_proba))
        results[f"{name}_pr_auc"] = float(average_precision_score(y, y_proba))
        results[f"{name}_n"] = len(y)
        results[f"{name}_pos"] = int(y.sum())
    results["n_features"] = len(common)
    results["threshold"] = threshold
    results["best_iteration"] = model.best_iteration
    return results


def main(quick: bool):
    tag = date.today().strftime("%Y%m%d")
    n_est = 200 if quick else 1000

    print("=" * 60)
    print("SPATIAL ABLATION STUDY")
    print("=" * 60)

    raw_path = OUTPUT_DIR / "raw_250k_20260228.parquet"
    if not raw_path.exists():
        raw_path = sorted(OUTPUT_DIR.glob("raw_*.parquet"))[-1]
    print(f"\n  Loading raw: {raw_path.name}")
    df_raw = pd.read_parquet(raw_path)
    print(f"  Shape: {df_raw.shape}")

    # Detect available buffer radii
    buffer_re = re.compile(r"defo_rate_(\d+)m_")
    available_radii = sorted({int(buffer_re.match(c).group(1))
                              for c in df_raw.columns if buffer_re.match(c)})
    print(f"  Available buffer radii: {available_radii}")

    all_results = []

    def build_train_val(df_raw_sub):
        """Build train and val from raw DataFrame."""
        df_train = rebuild_features_dataset(
            df_raw_sub, TRAIN_FEATURE_YEARS, TRAIN_PREDICTION_YEARS, feature_window=4)
        df_train["split"] = "train"
        df_val = rebuild_features_dataset(
            df_raw_sub, VAL_FEATURE_YEARS, VAL_PREDICTION_YEARS, feature_window=4)
        df_val["split"] = "val"
        return df_train, df_val

    # ═══════════════════════════════════════════════════════════════════════
    # 3A: Buffer radius combinations
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("3A — BUFFER RADIUS ABLATION")
    print("=" * 60)

    combos = []
    for r in available_radii:
        combos.append(([r], f"only_{r}m"))
    if len(available_radii) >= 2:
        for i, r1 in enumerate(available_radii):
            for r2 in available_radii[i+1:]:
                combos.append(([r1, r2], f"{r1}m+{r2}m"))
    if len(available_radii) > 2:
        combos.append((available_radii, "all_radii"))

    for radii, label in combos:
        print(f"\n--- {label} ---")
        t0 = time.time()

        buffer_cols_to_keep = [c for c in df_raw.columns
                               if not buffer_re.match(c) or
                               int(buffer_re.match(c).group(1)) in radii]
        df_raw_filtered = df_raw[buffer_cols_to_keep].copy()

        df_train, df_val = build_train_val(df_raw_filtered)
        df_train, train_feat = prepare_features(df_train)
        df_val, val_feat = prepare_features(df_val)

        train_core = core_cols(train_feat)
        val_core = core_cols(val_feat)
        core = [c for c in train_core if c in val_core]
        print(f"  {len(core)} features")

        result = train_and_eval(df_train, df_val, core, n_est=n_est)
        result["experiment"] = "buffer_radius"
        result["radii"] = radii
        result["label"] = label
        elapsed = time.time() - t0

        print(f"  Val AUC-ROC: {result['val_auc_roc']:.4f}  "
              f"PR-AUC: {result['val_pr_auc']:.4f}  ({elapsed:.0f}s)")
        all_results.append(result)

    # ═══════════════════════════════════════════════════════════════════════
    # 3B: Sampling density
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("3B — SAMPLING DENSITY ABLATION")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n_full = len(df_raw)

    for n_points in [50_000, 100_000, n_full]:
        label = f"{n_points//1000}K" if n_points < n_full else f"{n_full//1000}K (full)"
        print(f"\n--- {label} points ---")
        t0 = time.time()

        if n_points < n_full:
            idx = rng.choice(n_full, n_points, replace=False)
            df_sub = df_raw.iloc[idx].copy().reset_index(drop=True)
        else:
            df_sub = df_raw.copy()

        df_train, df_val = build_train_val(df_sub)
        df_train, train_feat = prepare_features(df_train)
        df_val, val_feat = prepare_features(df_val)
        train_core = core_cols(train_feat)
        val_core = core_cols(val_feat)
        core = [c for c in train_core if c in val_core]
        print(f"  Train: {len(df_train):,}, Val: {len(df_val):,}, {len(core)} features")

        result = train_and_eval(df_train, df_val, core, n_est=n_est)
        result["experiment"] = "sampling_density"
        result["n_points"] = n_points
        result["label"] = label
        elapsed = time.time() - t0

        print(f"  Val AUC-ROC: {result['val_auc_roc']:.4f}  "
              f"PR-AUC: {result['val_pr_auc']:.4f}  ({elapsed:.0f}s)")
        all_results.append(result)

    # ═══════════════════════════════════════════════════════════════════════
    # 3C: Spatial decay analysis
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("3C — SPATIAL DECAY ANALYSIS")
    print("=" * 60)

    if len(available_radii) >= 2:
        # For each available radius, compute mean defo_rate for the most recent year
        latest_yr = TRAIN_FEATURE_YEARS[-1]
        decay_data = {}
        for r in available_radii:
            col = f"defo_rate_{r}m_{latest_yr}"
            if col in df_raw.columns:
                vals = df_raw[col].dropna()
                decay_data[r] = {
                    "mean": float(vals.mean()),
                    "median": float(vals.median()),
                    "std": float(vals.std()),
                    "pct_nonzero": float((vals > 0).mean() * 100),
                }
                print(f"  r={r}m: mean={vals.mean():.4f}, median={vals.median():.4f}, "
                      f"nonzero={decay_data[r]['pct_nonzero']:.1f}%")

        if decay_data:
            # Plot decay curve
            radii_sorted = sorted(decay_data.keys())
            means = [decay_data[r]["mean"] for r in radii_sorted]
            stds = [decay_data[r]["std"] for r in radii_sorted]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.errorbar(radii_sorted, means, yerr=stds, fmt="o-", capsize=5,
                        color="steelblue", linewidth=2, markersize=8)
            ax.set_xlabel("Buffer Radius (m)")
            ax.set_ylabel(f"Mean Deforestation Rate ({latest_yr})")
            ax.set_title("Spatial Decay of Deforestation Contagion")
            ax.set_xscale("log")
            for r, m in zip(radii_sorted, means):
                ax.annotate(f"{m:.4f}", (r, m), textcoords="offset points",
                            xytext=(0, 12), ha="center", fontsize=10)
            plt.tight_layout()
            path = OUTPUT_DIR / f"spatial_decay_{tag}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {path.name}")

            all_results.append({
                "experiment": "spatial_decay",
                "decay_data": decay_data,
            })
    else:
        print("  Need ≥2 buffer radii for decay analysis")

    # ═══════════════════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════════════════
    results_path = OUTPUT_DIR / f"spatial_ablation_{tag}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path.name}")

    # Summary
    print("\n" + "=" * 60)
    print("SPATIAL ABLATION COMPLETE")
    print("=" * 60)

    print("\n  3A — Buffer Radius:")
    for r in all_results:
        if r.get("experiment") == "buffer_radius":
            print(f"    {r['label']:20s}: AUC-ROC={r['val_auc_roc']:.4f}  "
                  f"PR-AUC={r['val_pr_auc']:.4f}  ({r['n_features']} feat)")

    print("\n  3B — Sampling Density:")
    for r in all_results:
        if r.get("experiment") == "sampling_density":
            print(f"    {r['label']:20s}: AUC-ROC={r['val_auc_roc']:.4f}  "
                  f"PR-AUC={r['val_pr_auc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spatial ablation study")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    main(args.quick)
