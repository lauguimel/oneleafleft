"""Temporal ablation study: window depth and encoding types.

Phase 2 of the triple ablation:
  2A — Feature window depth: {2, 3, 4} years
  2B — Temporal encoding types: lags only, lags+summaries, lags+deltas, full
  2C — Interaction: temporal depth × feature group

Uses core features only (hansen + spatial + infra) for fair comparison.

Usage:
    conda activate deforest
    python scripts/temporal_ablation.py
    python scripts/temporal_ablation.py --dataset data/raw_250k_20260228.parquet
    python scripts/temporal_ablation.py --quick
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
from ablation_study import resolve_group, FEATURE_GROUPS
from gee_extraction import rebuild_features_dataset

OUTPUT_DIR = PROJECT_DIR / "data"
CORE_GROUPS = ["hansen", "spatial", "infra"]

# Airtight split — train NEVER sees 2023, val predicts 2024
TRAIN_FEATURE_YEARS = list(range(2016, 2023))  # [2016..2022]
TRAIN_PREDICTION_YEARS = [2020, 2021, 2022]
VAL_FEATURE_YEARS = list(range(2017, 2024))    # [2017..2023]
VAL_PREDICTION_YEARS = [2024]


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply same feature prep as train_xgboost.load_dataset (minus file loading)."""
    # Drop leaky global-anom columns
    leaky_anom = [c for c in df.columns if _GLOBAL_ANOM_RE.search(c) and "proxy" not in c]
    if leaky_anom:
        df = df.drop(columns=leaky_anom)

    # Add window summaries
    df = add_window_summaries(df)

    # Exclude global summaries
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

    # Drop structural NaN columns (>45%)
    nan_pct = df[feature_cols].isna().mean() * 100
    structural_nan = [c for c in feature_cols if nan_pct[c] > 45]
    feature_cols = [c for c in feature_cols if c not in structural_nan]

    return df, feature_cols


def filter_core_cols(feature_cols: list[str]) -> list[str]:
    """Keep only hansen + spatial + infra features."""
    core = []
    for g in CORE_GROUPS:
        core.extend(resolve_group(g, feature_cols))
    return list(dict.fromkeys(core))


def filter_encoding(feature_cols: list[str], encoding: str) -> list[str]:
    """Filter columns by temporal encoding type.

    Encoding modes:
      'lags_only':    keep Lag* cols only (no d1, wmean, wtrend, wanom)
      'lags_summaries': keep Lag* + wmean + wtrend (no d1, wanom)
      'lags_deltas':  keep Lag* + d1_* (no wmean, wtrend, wanom)
      'full':         keep everything (current default)
    """
    static = []     # no Lag/d1/wmean/wtrend/wanom
    lags = []       # *_Lag\d+
    deltas = []     # *_d1_* or *_d3_*
    wmean = []      # *_wmean
    wtrend = []     # *_wtrend
    wanom = []      # *_wanom_*

    lag_re = re.compile(r"_Lag\d+$")
    d_re = re.compile(r"_d\d+_")
    wanom_re = re.compile(r"_wanom_")

    for c in feature_cols:
        if wanom_re.search(c):
            wanom.append(c)
        elif c.endswith("_wmean"):
            wmean.append(c)
        elif c.endswith("_wtrend"):
            wtrend.append(c)
        elif d_re.search(c):
            deltas.append(c)
        elif lag_re.search(c):
            lags.append(c)
        else:
            static.append(c)

    if encoding == "lags_only":
        return static + lags
    elif encoding == "lags_summaries":
        return static + lags + wmean + wtrend
    elif encoding == "lags_deltas":
        return static + lags + deltas
    elif encoding == "full":
        return feature_cols
    else:
        raise ValueError(f"Unknown encoding: {encoding}")


def train_and_eval(train_df, val_df, feature_cols, n_est=1000):
    """Train XGBoost on given features and return metrics dict."""
    # Use only columns present in both
    common = [c for c in feature_cols if c in train_df.columns and c in val_df.columns]

    X_train = train_df[common].values
    y_train = train_df["target"].values.astype(int)
    X_val = val_df[common].values
    y_val = val_df["target"].values.astype(int)

    if y_train.sum() == 0 or y_val.sum() == 0:
        return {"val_auc_roc": np.nan, "val_pr_auc": np.nan, "n_features": len(common)}

    spw = compute_scale_pos_weight(y_train)
    model = train(X_train, y_train, X_val, y_val,
                  scale_pos_weight=spw, n_estimators=n_est)
    threshold = optimal_threshold(model, X_val, y_val)

    from sklearn.metrics import roc_auc_score, average_precision_score
    results = {}
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val)]:
        y_proba = model.predict_proba(X)[:, 1]
        results[f"{name}_auc_roc"] = float(roc_auc_score(y, y_proba))
        results[f"{name}_pr_auc"] = float(average_precision_score(y, y_proba))
        results[f"{name}_n"] = len(y)
        results[f"{name}_pos"] = int(y.sum())

    results["n_features"] = len(common)
    results["threshold"] = threshold
    results["best_iteration"] = model.best_iteration
    return results


def main(raw_path: Path | None, quick: bool):
    tag = date.today().strftime("%Y%m%d")
    n_est = 200 if quick else 1000

    print("=" * 60)
    print("TEMPORAL ABLATION STUDY")
    print("=" * 60)

    # ── Load raw data ───────────────────────────────────────────────────────
    if raw_path is None:
        candidates = sorted(OUTPUT_DIR.glob("raw_*.parquet"))
        if not candidates:
            print("ERROR: No raw_*.parquet found")
            sys.exit(1)
        raw_path = candidates[-1]

    print(f"\n  Loading raw: {raw_path.name}")
    df_raw = pd.read_parquet(raw_path)
    print(f"  Shape: {df_raw.shape}")

    all_results = []

    def build_train_val(window):
        """Rebuild train and val DataFrames for a given window depth."""
        df_train = rebuild_features_dataset(
            df_raw, years=TRAIN_FEATURE_YEARS,
            prediction_years=TRAIN_PREDICTION_YEARS,
            feature_window=window,
        )
        df_train["split"] = "train"

        df_val = rebuild_features_dataset(
            df_raw, years=VAL_FEATURE_YEARS,
            prediction_years=VAL_PREDICTION_YEARS,
            feature_window=window,
        )
        df_val["split"] = "val"
        return df_train, df_val

    # ═══════════════════════════════════════════════════════════════════════
    # 2A: Window depth ablation
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("2A — WINDOW DEPTH ABLATION")
    print("=" * 60)

    for window in [1, 2, 3, 4]:
        print(f"\n--- feature_window={window} ---")
        t0 = time.time()

        df_train, df_val = build_train_val(window)
        print(f"  Train: {len(df_train):,} rows, Val: {len(df_val):,} rows")

        df_train, train_feat = prepare_features(df_train)
        df_val, val_feat = prepare_features(df_val)
        core_train = filter_core_cols(train_feat)
        core_val = filter_core_cols(val_feat)
        core_cols = list(dict.fromkeys(c for c in core_train if c in core_val))
        print(f"  Core features: {len(core_cols)}")

        result = train_and_eval(df_train, df_val, core_cols, n_est=n_est)
        result["experiment"] = "window_depth"
        result["window"] = window
        elapsed = time.time() - t0

        print(f"  Val AUC-ROC: {result['val_auc_roc']:.4f}  "
              f"PR-AUC: {result['val_pr_auc']:.4f}  "
              f"({elapsed:.0f}s)")
        all_results.append(result)

    # ═══════════════════════════════════════════════════════════════════════
    # 2B: Encoding type ablation (at window=4, the current default)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("2B — ENCODING TYPE ABLATION (window=4)")
    print("=" * 60)

    # Reuse window=4 dataset
    df_train_w4, df_val_w4 = build_train_val(4)
    df_train_w4, train_feat_w4 = prepare_features(df_train_w4)
    df_val_w4, val_feat_w4 = prepare_features(df_val_w4)
    core_train_w4 = filter_core_cols(train_feat_w4)
    core_val_w4 = filter_core_cols(val_feat_w4)
    core_cols_w4 = list(dict.fromkeys(c for c in core_train_w4 if c in core_val_w4))

    for encoding in ["lags_only", "lags_summaries", "lags_deltas", "full"]:
        print(f"\n--- encoding={encoding} ---")
        t0 = time.time()

        filtered = filter_encoding(core_cols_w4, encoding)
        filtered = [c for c in filtered if c in df_train_w4.columns and c in df_val_w4.columns]
        print(f"  Features: {len(filtered)}")

        result = train_and_eval(df_train_w4, df_val_w4, filtered, n_est=n_est)
        result["experiment"] = "encoding_type"
        result["encoding"] = encoding
        elapsed = time.time() - t0

        print(f"  Val AUC-ROC: {result['val_auc_roc']:.4f}  "
              f"PR-AUC: {result['val_pr_auc']:.4f}  "
              f"({elapsed:.0f}s)")
        all_results.append(result)

    # ═══════════════════════════════════════════════════════════════════════
    # 2C: Interaction — window depth × feature group
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("2C — INTERACTION: WINDOW DEPTH × FEATURE GROUP")
    print("=" * 60)

    for window in [1, 2, 3, 4]:
        df_train_w, df_val_w = build_train_val(window)
        df_train_w, train_feat_w = prepare_features(df_train_w)
        df_val_w, val_feat_w = prepare_features(df_val_w)

        for group in CORE_GROUPS:
            print(f"\n--- window={window}, group={group} ---")
            t0 = time.time()

            train_group = resolve_group(group, train_feat_w)
            val_group = resolve_group(group, val_feat_w)
            group_cols = [c for c in train_group if c in val_group]
            if not group_cols:
                print(f"  No features for {group}")
                continue

            result = train_and_eval(df_train_w, df_val_w, group_cols, n_est=n_est)
            result["experiment"] = "interaction"
            result["window"] = window
            result["group"] = group
            elapsed = time.time() - t0

            print(f"  {len(group_cols)} features → "
                  f"Val AUC-ROC: {result['val_auc_roc']:.4f}  "
                  f"PR-AUC: {result['val_pr_auc']:.4f}  "
                  f"({elapsed:.0f}s)")
            all_results.append(result)

    # ═══════════════════════════════════════════════════════════════════════
    # Save results + plot
    # ═══════════════════════════════════════════════════════════════════════
    results_path = OUTPUT_DIR / f"temporal_ablation_{tag}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {results_path.name}")

    # ── Plot 2A: window depth ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    depth_results = [r for r in all_results if r["experiment"] == "window_depth"]
    windows = [r["window"] for r in depth_results]
    auc_roc = [r["val_auc_roc"] for r in depth_results]
    pr_auc = [r["val_pr_auc"] for r in depth_results]

    axes[0].bar(range(len(windows)), auc_roc, tick_label=windows, color="steelblue")
    axes[0].set_xlabel("Feature Window (years)")
    axes[0].set_ylabel("Val AUC-ROC")
    axes[0].set_title("Window Depth vs AUC-ROC")
    axes[0].set_ylim(min(auc_roc) - 0.02, max(auc_roc) + 0.02)
    for i, v in enumerate(auc_roc):
        axes[0].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=10)

    axes[1].bar(range(len(windows)), pr_auc, tick_label=windows, color="darkorange")
    axes[1].set_xlabel("Feature Window (years)")
    axes[1].set_ylabel("Val PR-AUC")
    axes[1].set_title("Window Depth vs PR-AUC")
    axes[1].set_ylim(min(pr_auc) - 0.005, max(pr_auc) + 0.005)
    for i, v in enumerate(pr_auc):
        axes[1].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=10)

    plt.suptitle("Temporal Ablation: Window Depth", fontsize=14)
    plt.tight_layout()
    path = OUTPUT_DIR / f"temporal_window_depth_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")

    # ── Plot 2B: encoding types ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    enc_results = [r for r in all_results if r["experiment"] == "encoding_type"]
    encodings = [r["encoding"] for r in enc_results]
    auc_roc_e = [r["val_auc_roc"] for r in enc_results]
    pr_auc_e = [r["val_pr_auc"] for r in enc_results]

    axes[0].barh(encodings, auc_roc_e, color="steelblue")
    axes[0].set_xlabel("Val AUC-ROC")
    axes[0].set_title("Encoding Type vs AUC-ROC")
    for i, v in enumerate(auc_roc_e):
        axes[0].text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=10)

    axes[1].barh(encodings, pr_auc_e, color="darkorange")
    axes[1].set_xlabel("Val PR-AUC")
    axes[1].set_title("Encoding Type vs PR-AUC")
    for i, v in enumerate(pr_auc_e):
        axes[1].text(v + 0.0005, i, f"{v:.4f}", va="center", fontsize=10)

    plt.suptitle("Temporal Ablation: Encoding Types (window=4)", fontsize=14)
    plt.tight_layout()
    path = OUTPUT_DIR / f"temporal_encoding_types_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")

    # ── Summary table ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEMPORAL ABLATION COMPLETE")
    print("=" * 60)
    print(f"\n  2A — Window Depth:")
    for r in depth_results:
        print(f"    window={r['window']}: AUC-ROC={r['val_auc_roc']:.4f}  "
              f"PR-AUC={r['val_pr_auc']:.4f}  ({r['n_features']} features)")
    print(f"\n  2B — Encoding Types:")
    for r in enc_results:
        print(f"    {r['encoding']:18s}: AUC-ROC={r['val_auc_roc']:.4f}  "
              f"PR-AUC={r['val_pr_auc']:.4f}  ({r['n_features']} features)")
    print(f"\n  2C — Interaction (window × group):")
    int_results = [r for r in all_results if r["experiment"] == "interaction"]
    for r in int_results:
        print(f"    w={r['window']}, {r['group']:8s}: AUC-ROC={r['val_auc_roc']:.4f}  "
              f"PR-AUC={r['val_pr_auc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal ablation study")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Path to raw_*.parquet (not features)")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    main(args.dataset, args.quick)
