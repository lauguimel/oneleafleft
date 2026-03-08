"""Ablation study: train XGBoost with different feature subsets.

Scenarios:
  - Each feature group alone          (9 individual runs)
  - Cumulative builds (hansen → all)  (8 incremental runs)
  - All features                      (1 run)

Usage:
    conda activate deforest
    python scripts/ablation_study.py                                    # auto-detect latest parquet
    python scripts/ablation_study.py --dataset data/features_250k_20260228.parquet
    python scripts/ablation_study.py --dataset data/features_250k_20260228.parquet --quick

Output:
    data/ablation_results_YYYYMMDD.json   — per-scenario metrics
    data/ablation_barplot_YYYYMMDD.png    — AUC-ROC and PR-AUC barplot
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Callable

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
    NON_FEATURE_COLS,
    _GLOBAL_ANOM_RE,
    _GLOBAL_SUMMARY_SUFFIXES,
)
from gee_extraction import rebuild_features_dataset

OUTPUT_DIR = PROJECT_DIR / "data"

# Airtight split
TRAIN_FEATURE_YEARS = list(range(2016, 2023))
TRAIN_PREDICTION_YEARS = [2020, 2021, 2022]
VAL_FEATURE_YEARS = list(range(2017, 2024))
VAL_PREDICTION_YEARS = [2024]


# ─── Feature group definitions ────────────────────────────────────────────────
# Each group is a list of predicates: col is included if ANY predicate matches.

def _sw(*prefixes: str) -> Callable[[str], bool]:
    """Startswith any of prefixes."""
    return lambda col: any(col.startswith(p) for p in prefixes)


def _exact(*names: str) -> Callable[[str], bool]:
    _s = set(names)
    return lambda col: col in _s


FEATURE_GROUPS: dict[str, list[Callable[[str], bool]]] = {
    "hansen": [
        _sw("cum_deforested"),
        _exact("treecover2000", "cum_loss_before"),
    ],
    "spatial": [_sw("defo_rate_")],   # Hansen buffer deforestation rates (contagion)
    "infra": [
        _exact(
            "dist_road_km", "dist_settlement_km",
            "in_protected", "dist_protected_km",
            "iucn_strict", "iucn_moderate", "iucn_sustainable", "iucn_not_reported",
            "pa_defo_rate", "pa_pressure_ring",
            "elevation", "slope",
        ),
    ],
    "pop": [_sw("pop_")],
    "ntl": [_sw("ntl_")],
    "climate": [
        _sw(
            "temperature_2m", "dry_days", "hot_days",
            "et_", "sm_", "extreme_rain_days",
        ),
    ],
    "precip": [_sw("precip_")],
    "governance": [_sw("WGI")],
    "economy": [_sw("WDI", "price_")],
    "country": [_sw("country_")],
}

CUMULATIVE_ORDER = [
    "hansen", "spatial", "infra", "pop", "ntl",
    "climate", "precip", "governance", "economy", "country",
]


def resolve_group(group: str, feature_cols: list[str]) -> list[str]:
    preds = FEATURE_GROUPS[group]
    return [c for c in feature_cols if any(p(c) for p in preds)]


def build_scenarios(feature_cols: list[str]) -> list[dict]:
    """Return list of {name, group_label, cols} dicts."""
    scenarios = []

    # ── Individual groups ─────────────────────────────────────────────────────
    for g in CUMULATIVE_ORDER:
        cols = resolve_group(g, feature_cols)
        scenarios.append({"name": f"only_{g}", "label": g, "cols": cols})

    # ── Cumulative builds (skip first = duplicate of only_hansen) ─────────────
    seen: set[str] = set()
    cumul: list[str] = []
    for g in CUMULATIVE_ORDER:
        for c in resolve_group(g, feature_cols):
            if c not in seen:
                cumul.append(c)
                seen.add(c)
        if g != CUMULATIVE_ORDER[0]:
            scenarios.append({
                "name": f"cumul_+{g}",
                "label": f"cumulative through {g}",
                "cols": list(cumul),
            })

    # ── Full model ────────────────────────────────────────────────────────────
    scenarios.append({"name": "all_features", "label": "all", "cols": feature_cols})

    return scenarios


def coverage_check(feature_cols: list[str]) -> None:
    """Print how many features each group covers (sanity check)."""
    covered: set[str] = set()
    print("\n  Group coverage:")
    for g in CUMULATIVE_ORDER:
        cols = resolve_group(g, feature_cols)
        covered.update(cols)
        print(f"    {g:<12} {len(cols):>4} features")
    uncovered = [c for c in feature_cols if c not in covered]
    print(f"    {'UNCOVERED':<12} {len(uncovered):>4} features")
    if uncovered:
        print(f"      {uncovered[:10]}")
    print(f"    {'TOTAL':<12} {len(feature_cols):>4} features")


def run_scenario(
    s: dict,
    train_df, val_df,
    n_estimators: int,
) -> dict | None:
    cols = s["cols"]
    if not cols:
        print(f"  [SKIP] {s['name']}: 0 features — group absent from dataset")
        return None

    # Filter to cols present in both train and val
    available = set(train_df.columns) & set(val_df.columns)
    cols = [c for c in cols if c in available]
    if not cols:
        print(f"  [SKIP] {s['name']}: 0 common features")
        return None

    print(f"\n{'─'*55}")
    print(f"  {s['name']}  ({len(cols)} features)")
    print(f"{'─'*55}")

    X_tr = train_df[cols].values;  y_tr = train_df["target"].values.astype(int)
    X_va = val_df[cols].values;    y_va = val_df["target"].values.astype(int)

    spw = compute_scale_pos_weight(y_tr)
    t0 = time.time()
    model = train(X_tr, y_tr, X_va, y_va,
                  scale_pos_weight=spw, n_estimators=n_estimators)
    elapsed = time.time() - t0
    print(f"  Trained {elapsed:.1f}s  best_iter={model.best_iteration}")

    vm = evaluate(model, X_va, y_va, "val")

    return {
        "name": s["name"],
        "label": s["label"],
        "n_features": len(cols),
        "best_iteration": model.best_iteration,
        "train_time_s": round(elapsed, 1),
        "val_auc_roc": vm["auc_roc"],
        "val_pr_auc": vm["auc_pr"],
    }


def print_table(results: list[dict]) -> None:
    header = f"{'Scenario':<30} {'N_feat':>7} {'Val AUC':>9} {'Val PR-AUC':>11}"
    sep = "─" * len(header)
    print(f"\n{sep}\nABLATION RESULTS\n{sep}")
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['name']:<30} {r['n_features']:>7} "
            f"{r['val_auc_roc']:>9.4f} "
            f"{r['val_pr_auc']:>11.4f}"
        )
    print(sep)


def plot_results(results: list[dict], path: Path) -> None:
    names    = [r["name"] for r in results]
    auc_rocs = [r["val_auc_roc"] for r in results]
    pr_aucs  = [r["val_pr_auc"]  for r in results]

    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.75), 6))

    b1 = ax.bar(x - w/2, auc_rocs, w, label="AUC-ROC (val)", color="steelblue")
    b2 = ax.bar(x + w/2, pr_aucs,  w, label="PR-AUC (val)",  color="coral")

    for b in (b1, b2):
        for bar in b:
            h = bar.get_height()
            if h > 0.005:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

    n_ind  = len(CUMULATIVE_ORDER)
    n_cum  = len(CUMULATIVE_ORDER) - 1
    ax.axvline(n_ind - 0.5,         color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(n_ind + n_cum - 0.5, color="gray", ls="--", lw=0.8, alpha=0.6)

    ymax = 1.08
    ax.text((n_ind-1)/2,              ymax-0.02, "Individual groups",  ha="center", fontsize=8, color="gray")
    ax.text(n_ind + (n_cum-1)/2,      ymax-0.02, "Cumulative builds",  ha="center", fontsize=8, color="gray")
    ax.text(n_ind + n_cum,            ymax-0.02, "Full",               ha="center", fontsize=8, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score on val set (pred_year=2024)")
    ax.set_title("Ablation Study — XGBoost Feature Groups\nDeforestation Prediction, Congo Basin 250K points")
    ax.legend(loc="upper right")
    ax.set_ylim(0, ymax)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Barplot: {path.name}")


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


def main(quick: bool) -> None:
    tag = date.today().strftime("%Y%m%d")
    n_est = 200 if quick else 1000

    print("=" * 60)
    print("ABLATION STUDY — Deforestation Prediction (airtight split)")
    print(f"  n_estimators={n_est}  quick={quick}")
    print("=" * 60)

    # Load raw data
    raw_path = OUTPUT_DIR / "raw_250k_20260228.parquet"
    if not raw_path.exists():
        raw_path = sorted(OUTPUT_DIR.glob("raw_*.parquet"))[-1]
    print(f"\n[1] Loading raw: {raw_path.name}")
    df_raw = pd.read_parquet(raw_path)
    print(f"  {len(df_raw):,} points, {df_raw.shape[1]} columns")

    # Build train and val from raw
    print("\n[2] Building airtight train/val...")
    df_train = rebuild_features_dataset(
        df_raw, TRAIN_FEATURE_YEARS, TRAIN_PREDICTION_YEARS, feature_window=4)
    df_train["split"] = "train"
    df_val = rebuild_features_dataset(
        df_raw, VAL_FEATURE_YEARS, VAL_PREDICTION_YEARS, feature_window=4)
    df_val["split"] = "val"
    print(f"  Train: {len(df_train):,} rows, Val: {len(df_val):,} rows")

    # Prepare features
    df_train, train_feat = prepare_features(df_train)
    df_val, val_feat = prepare_features(df_val)
    feature_cols = sorted(set(train_feat) & set(val_feat))
    print(f"  {len(feature_cols)} common feature columns")
    coverage_check(feature_cols)

    print("\n[3] Building scenarios...")
    scenarios = build_scenarios(feature_cols)
    for s in scenarios:
        status = "SKIP" if not s["cols"] else f"{len(s['cols']):>4} feat"
        print(f"  {s['name']:<30} {status}")
    print(f"\n  {len(scenarios)} total scenarios")

    print("\n[4] Running ablation...")
    results, skipped = [], []
    for i, s in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}]", end="")
        r = run_scenario(s, df_train, df_val, n_est)
        if r is None:
            skipped.append(s["name"])
        else:
            results.append(r)

    print_table(results)
    if skipped:
        print(f"\n  Skipped ({len(skipped)}): {skipped}")

    json_path = OUTPUT_DIR / f"ablation_results_{tag}.json"
    with open(json_path, "w") as f:
        json.dump({
            "raw": str(raw_path),
            "n_estimators": n_est,
            "quick": quick,
            "n_run": len(results),
            "n_skipped": len(skipped),
            "skipped": skipped,
            "results": results,
        }, f, indent=2)
    print(f"\n  JSON: {json_path.name}")

    plot_results(results, OUTPUT_DIR / f"ablation_barplot_{tag}.png")

    print("\n" + "=" * 60)
    print("ABLATION COMPLETE")
    print(f"  Best individual group: "
          + max((r for r in results if r["name"].startswith("only_")),
                key=lambda r: r["val_auc_roc"])["name"])
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="200 trees instead of 1000")
    args = parser.parse_args()
    main(args.quick)
