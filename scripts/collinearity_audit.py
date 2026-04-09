"""Collinearity audit (Phase A): correlations, VIF, clustering.

Computes Pearson and Spearman correlation matrices, per-family VIF,
hierarchical clustering on 1 - |rho|, and extracts cluster memberships at
|rho| > 0.9 and > 0.7. Saves heatmaps and CSV artifacts to
``results/collinearity/`` and writes a summary to
``results/collinearity_audit.md``.

Usage:
    python scripts/collinearity_audit.py [--data PATH] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from src.evaluation.collinearity import (
    annotate_pairs,
    classify_feature_family,
    cluster_features,
    compute_correlations,
    compute_vif,
    extract_top_pairs,
    select_representatives,
)
from src.evaluation.family_ablation import (
    compute_contagion_velocity,
    family_ablation,
)

DEFAULT_XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "verbosity": 0,
}

DEFAULT_DATA = "data/train_test/features_traintest_20260307.parquet"


def _save_heatmap(corr: pd.DataFrame, path: Path, title: str) -> None:
    n = corr.shape[0]
    fig_size = max(6, min(20, n * 0.25))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        xticklabels=(n <= 40),
        yticklabels=(n <= 40),
        cbar_kws={"shrink": 0.6},
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_dendrogram(corr: pd.DataFrame, path: Path, title: str) -> None:
    if corr.shape[0] < 2:
        return
    dist = 1.0 - corr.abs().values
    dist = np.nan_to_num(dist, nan=1.0, posinf=1.0, neginf=1.0)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2.0, 0.0, 1.0)
    Z = linkage(squareform(dist, checks=False), method="average")
    fig, ax = plt.subplots(figsize=(max(8, corr.shape[0] * 0.15), 6))
    dendrogram(Z, labels=list(corr.columns), leaf_rotation=90, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("1 - |rho|")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    parser.add_argument("--out", type=str, default="results/collinearity")
    parser.add_argument("--max-vif-per-family", type=int, default=60,
                        help="Cap features per family for VIF (cost control).")
    parser.add_argument("--ablation", action="store_true",
                        help="Run family ablation XGBoost study.")
    parser.add_argument("--velocity", action="store_true",
                        help="Compute contagion velocity stats.")
    parser.add_argument("--test-data", type=str, default=None,
                        help="Test parquet for ablation (required with --ablation).")
    parser.add_argument("--target", type=str, default="target")
    parser.add_argument("--params", type=str, default=None,
                        help="JSON file with XGBoost hyperparameters.")
    parser.add_argument("--year-col", type=str, default="year")
    parser.add_argument("--x-col", type=str, default="x_m")
    parser.add_argument("--y-col", type=str, default="y_m")
    parser.add_argument("--sample-n", type=int, default=None,
                        help="Subsample N rows before computing correlations (cost control).")
    parser.add_argument("--skip-spearman", action="store_true",
                        help="Skip Spearman (slowest step on full data).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-unique", type=int, default=5,
                        help="Drop features with fewer than N unique values "
                             "(filters constants + commodity-price-like artefacts). "
                             "Default 5 keeps sparse contagion features.")
    parser.add_argument("--protect", type=str, default="target,prediction_year,lossyear",
                        help="Comma-separated columns never dropped by min-unique filter.")
    parser.add_argument("--top-per-family", type=int, default=20,
                        help="Top-N pairs per family pair to report.")
    args = parser.parse_args()

    if args.params and Path(args.params).exists():
        xgb_params = json.loads(Path(args.params).read_text())
    else:
        xgb_params = DEFAULT_XGB_PARAMS

    data_path = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[collinearity] loading {data_path}", flush=True)
    df_full = pd.read_parquet(data_path)
    print(f"[collinearity] loaded {len(df_full)} rows × {df_full.shape[1]} cols", flush=True)
    if args.sample_n and len(df_full) > args.sample_n:
        df = df_full.sample(n=args.sample_n, random_state=args.seed).reset_index(drop=True)
        print(f"[collinearity] subsampled to {len(df)} rows for correlation step", flush=True)
    else:
        df = df_full
    numeric = df.select_dtypes(include=[np.number])
    n0 = numeric.shape[1]
    # Drop all-NaN, zero-variance, and quasi-constant columns.
    numeric = numeric.dropna(axis=1, how="all")
    numeric = numeric.loc[:, numeric.std(ddof=0) > 0]
    protected = {c.strip() for c in args.protect.split(",") if c.strip()}
    if args.min_unique > 1:
        nunique = numeric.nunique(dropna=True)
        keep = set(nunique[nunique >= args.min_unique].index) | (protected & set(numeric.columns))
        dropped = sorted(set(numeric.columns) - keep)
        numeric = numeric[[c for c in numeric.columns if c in keep]]
        if dropped:
            (out_dir / "dropped_low_variance.txt").write_text(
                "\n".join(dropped) + "\n"
            )
            print(f"[collinearity] dropped {len(dropped)} cols with <{args.min_unique} "
                  f"unique values (e.g. {dropped[:5]})", flush=True)
    print(f"[collinearity] numeric features: {numeric.shape[1]} (from {n0})", flush=True)

    # Family classification.
    families: dict[str, list[str]] = defaultdict(list)
    for col in numeric.columns:
        families[classify_feature_family(col)].append(col)
    fam_summary = pd.Series({k: len(v) for k, v in families.items()}, name="n_features")
    fam_summary.to_csv(out_dir / "family_counts.csv")
    print(f"[collinearity] families: {fam_summary.to_dict()}")

    # Global correlations.
    print("[collinearity] computing Pearson correlation...", flush=True)
    pearson = compute_correlations(numeric, method="pearson")
    pearson.to_csv(out_dir / "corr_pearson.csv")
    print("[collinearity]   pearson done", flush=True)

    if args.skip_spearman:
        spearman = pearson
        print("[collinearity] skipping Spearman", flush=True)
    else:
        print("[collinearity] computing Spearman correlation...", flush=True)
        spearman = compute_correlations(numeric, method="spearman")
        spearman.to_csv(out_dir / "corr_spearman.csv")
        print("[collinearity]   spearman done", flush=True)

    # Top pairs.
    top_pearson = extract_top_pairs(pearson, n=100)
    top_pearson.to_csv(out_dir / "top_pairs_pearson.csv", index=False)
    top_spearman = extract_top_pairs(spearman, n=100)
    top_spearman.to_csv(out_dir / "top_pairs_spearman.csv", index=False)
    print("[collinearity] top pairs extracted", flush=True)

    # Per-family heatmaps + VIF.
    vif_records = []
    for fam, cols in families.items():
        if len(cols) < 2:
            continue
        print(f"[collinearity] family={fam} ({len(cols)} cols): heatmap+VIF...", flush=True)
        sub = numeric[cols]
        fam_corr = compute_correlations(sub, method="pearson")
        _save_heatmap(fam_corr, out_dir / f"heatmap_{fam}.png",
                      f"Pearson correlation — {fam} (n={len(cols)})")
        _save_dendrogram(fam_corr, out_dir / f"dendrogram_{fam}.png",
                         f"Hierarchical clustering — {fam}")

        vif_cols = cols[: args.max_vif_per_family]
        try:
            vif = compute_vif(sub[vif_cols].dropna())
            vif_df = vif.reset_index()
            vif_df.columns = ["feature", "VIF"]
            vif_df["family"] = fam
            vif_records.append(vif_df)
        except Exception as exc:
            print(f"[collinearity] VIF failed for {fam}: {exc}")

    if vif_records:
        vif_all = pd.concat(vif_records, ignore_index=True)
        vif_all.sort_values("VIF", ascending=False, inplace=True)
        vif_all.to_csv(out_dir / "vif_per_family.csv", index=False)

    # Global clusters at two thresholds.
    clusters_90 = cluster_features(pearson, threshold=0.9)
    clusters_70 = cluster_features(pearson, threshold=0.7)
    pd.DataFrame(
        [(cid, f) for cid, members in clusters_90.items() for f in members],
        columns=["cluster_id", "feature"],
    ).to_csv(out_dir / "clusters_rho_0.9.csv", index=False)
    pd.DataFrame(
        [(cid, f) for cid, members in clusters_70.items() for f in members],
        columns=["cluster_id", "feature"],
    ).to_csv(out_dir / "clusters_rho_0.7.csv", index=False)

    # Global heatmap (if feasible).
    if numeric.shape[1] <= 200:
        _save_heatmap(pearson, out_dir / "heatmap_all.png",
                      f"Pearson correlation (all, n={numeric.shape[1]})")

    # Representative selection (Phase B) on |rho| > 0.9 clusters.
    representatives = select_representatives(clusters_90, pearson)
    decorr_path = Path("data/features_decorrelated.txt")
    decorr_path.parent.mkdir(parents=True, exist_ok=True)
    decorr_path.write_text("\n".join(representatives) + "\n")
    print(f"[collinearity] wrote {len(representatives)} representatives to {decorr_path}")

    # Annotate top-50 global pairs.
    top50_annotated = annotate_pairs(extract_top_pairs(pearson, n=50))
    top50_annotated.to_csv(out_dir / "top_pairs_annotated.csv", index=False)

    # Top-N per family pair (much more useful than global top).
    all_pairs = annotate_pairs(extract_top_pairs(pearson, n=10000))
    by_family_pair = []
    for (fa, fb), grp in all_pairs.groupby(["family_a", "family_b"]):
        by_family_pair.append(grp.head(args.top_per_family))
    top_by_pair = pd.concat(by_family_pair, ignore_index=True) if by_family_pair else pd.DataFrame()
    top_by_pair.to_csv(out_dir / "top_pairs_by_family.csv", index=False)
    # Within-family contagion-only top (the question that matters most).
    contagion_top = all_pairs[
        (all_pairs["family_a"] == "contagion") & (all_pairs["family_b"] == "contagion")
    ].head(30)
    contagion_top.to_csv(out_dir / "top_pairs_contagion.csv", index=False)
    # Cross-family pop x night_lights (Marc's example).
    pop_ntl = all_pairs[
        ((all_pairs["family_a"] == "demography") & (all_pairs["family_b"] == "night_lights"))
        | ((all_pairs["family_a"] == "night_lights") & (all_pairs["family_b"] == "demography"))
    ].head(30)
    pop_ntl.to_csv(out_dir / "top_pairs_pop_ntl.csv", index=False)

    # Summary markdown.
    summary_path = Path("results/collinearity_audit.md")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    n_redundant_90 = sum(1 for m in clusters_90.values() if len(m) > 1)
    n_redundant_70 = sum(1 for m in clusters_70.values() if len(m) > 1)
    lines = [
        "# Collinearity audit — Phase A",
        "",
        f"- Data: `{data_path}`",
        f"- Numeric features: **{numeric.shape[1]}**",
        f"- Samples: **{len(numeric)}**",
        "",
        "## Family counts",
        "",
        fam_summary.to_frame().to_markdown(),
        "",
        "## Clustering summary",
        "",
        f"- Clusters at |rho| > 0.9: **{len(clusters_90)}** "
        f"({n_redundant_90} multi-feature)",
        f"- Clusters at |rho| > 0.7: **{len(clusters_70)}** "
        f"({n_redundant_70} multi-feature)",
        "",
        "## Top 10 correlated pairs (Pearson)",
        "",
        top_pearson.head(10).to_markdown(index=False),
        "",
        "## Top contagion×contagion pairs (the central question)",
        "",
        contagion_top.head(20).to_markdown(index=False) if len(contagion_top) else "_no contagion pairs_",
        "",
        "## Top demography×night_lights pairs (Marc's example)",
        "",
        pop_ntl.head(20).to_markdown(index=False) if len(pop_ntl) else "_no such pairs_",
        "",
        "## Top 50 annotated pairs (global)",
        "",
        top50_annotated.to_markdown(index=False),
        "",
        "## Representatives (decorrelated feature set)",
        "",
        f"- Count: **{len(representatives)}**",
        f"- File: `{decorr_path}`",
        "",
        "Artifacts in `results/collinearity/`.",
        "",
    ]
    # --- Family ablation (optional) ---
    if args.ablation:
        print("[collinearity] running family ablation...", flush=True)
        if args.test_data:
            df_train = df_full
            df_test = pd.read_parquet(args.test_data)
        elif "split" in df_full.columns:
            df_train = df_full[df_full["split"] == "train"].reset_index(drop=True)
            df_test = df_full[df_full["split"] == "test"].reset_index(drop=True)
            print(f"[collinearity]   split from col: train={len(df_train)} test={len(df_test)}",
                  flush=True)
        else:
            raise ValueError("--ablation requires --test-data or a 'split' column.")
        fams = [f for f in families.keys() if f != "other"]
        ablation_df = family_ablation(
            df_train, df_test, args.target, classify_feature_family, fams, xgb_params,
        )
        ablation_df.to_csv(out_dir / "family_ablation.csv", index=False)
        lines += [
            "## Family ablation (XGBoost, PR-AUC)",
            "",
            ablation_df.to_markdown(index=False),
            "",
        ]

    # --- Contagion velocity (optional) ---
    if args.velocity and all(c in df.columns for c in
                              [args.year_col, args.x_col, args.y_col, args.target]):
        print("[collinearity] computing contagion velocity...")
        vel = compute_contagion_velocity(
            df, args.year_col, args.x_col, args.y_col, args.target,
        )
        pd.Series(vel).to_csv(out_dir / "contagion_velocity.csv")
        lines += [
            "## Contagion velocity (metres, year-over-year NN)",
            "",
            pd.Series(vel).to_frame("value").to_markdown(),
            "",
        ]

    summary_path.write_text("\n".join(lines))
    print(f"[collinearity] wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
