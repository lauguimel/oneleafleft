"""
paper_figures.py — Generate publication-quality figures for ERL manuscript.

Produces:
  paper/figures/fig1_study_area.{pdf,png}   + paper/figures/data/fig1_*.csv
  paper/figures/fig2_ablation.{pdf,png}     + paper/figures/data/fig2_*.csv
  paper/figures/fig3_lift_shap.{pdf,png}    + paper/figures/data/fig3_*.csv

Run: conda run -n deforest python scripts/paper_figures.py
"""

import json
import os
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PAPER_FIG = ROOT / "paper" / "figures"
DATA_DIR = ROOT / "paper" / "figures" / "data"
PAPER_FIG.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style — modern palette, larger fonts
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         13,
    "axes.labelsize":    13,
    "axes.titlesize":    14,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   11,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Modern color palette
PALETTE = {
    "indigo":  "#4F46E5",   # deep blue — primary bars
    "teal":    "#0891B2",   # vibrant teal — spatial
    "coral":   "#F43F5E",   # coral/rose — PR-AUC line + deforested dots
    "amber":   "#F59E0B",   # warm amber — encoding markers
    "emerald": "#10B981",   # green — infra/pop group
    "violet":  "#7C3AED",   # violet — accent
    "slate":   "#64748B",   # grey — baselines, neutral
    "red":     "#EF4444",   # bright red — deforested points
    "bg":      "#F1F5F9",   # very light blue-grey — panel background hint
}


# ===========================================================================
# FIG 1 — Study area (scatter) + Predicted risk map
# ===========================================================================
def make_fig1():
    print("Fig 1: study area + risk map...")

    pred = pd.read_parquet(ROOT / "data" / "app" / "predictions_val.parquet")
    pred = pred[["lon", "lat", "target", "proba"]].copy()

    # Country boundaries
    world = gpd.read_file(ROOT / "data" / "boundaries" / "ne_110m_countries.gpkg")
    lon_min = pred["lon"].min() - 0.5
    lon_max = pred["lon"].max() + 0.5
    lat_min = pred["lat"].min() - 0.5
    lat_max = pred["lat"].max() + 0.5

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # ---- Panel A: scatter of ALL points, red overlay for positives ----
    ax = axes[0]
    world.clip([lon_min, lat_min, lon_max, lat_max]).plot(
        ax=ax, facecolor="#E2E8F0", edgecolor="#94A3B8", linewidth=0.6
    )
    neg = pred[pred["target"] == 0]
    pos = pred[pred["target"] == 1]

    # Subsample negatives for speed (keep all positives)
    rng = np.random.default_rng(42)
    neg_idx = rng.choice(len(neg), size=min(60000, len(neg)), replace=False)
    neg_sub = neg.iloc[neg_idx]

    ax.scatter(
        neg_sub["lon"], neg_sub["lat"],
        s=2, c="#94A3B8", alpha=0.20, linewidths=0,
        rasterized=True, label="No loss (sample)",
    )
    ax.scatter(
        pos["lon"], pos["lat"],
        s=10, c=PALETTE["coral"], alpha=0.85, linewidths=0,
        rasterized=True, label="Deforested 2024",
        zorder=5,
    )
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("(a) Validation sample (2024)", loc="left",
                 fontsize=14, fontweight="bold")
    ax.legend(markerscale=2.5, frameon=False, fontsize=10,
              loc="upper left", handletextpad=0.4)

    # ---- Panel B: predicted risk map ----
    ax = axes[1]
    idx_b = rng.choice(len(pred), size=min(50000, len(pred)), replace=False)
    sub = pred.iloc[idx_b].sort_values("proba")   # low risk first → high risk on top

    world.clip([lon_min, lat_min, lon_max, lat_max]).plot(
        ax=ax, facecolor="#E2E8F0", edgecolor="#94A3B8", linewidth=0.6
    )
    sc = ax.scatter(
        sub["lon"], sub["lat"],
        c=sub["proba"],
        cmap="YlOrRd",
        s=2,
        vmin=0, vmax=1,
        alpha=0.75,
        linewidths=0,
        rasterized=True,
    )
    cb = plt.colorbar(sc, ax=ax, label="Predicted risk", fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=10)
    cb.set_label("Predicted risk", fontsize=11)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("")
    ax.set_title("(b) Predicted deforestation risk", loc="left",
                 fontsize=14, fontweight="bold")

    plt.tight_layout(w_pad=2.5)
    fig.savefig(PAPER_FIG / "fig1_study_area.pdf")
    fig.savefig(PAPER_FIG / "fig1_study_area.png", dpi=300)
    plt.close(fig)

    pred[["lon", "lat", "target", "proba"]].to_csv(
        DATA_DIR / "fig1_predictions.csv", index=False
    )
    print("  -> fig1_study_area.pdf + data/fig1_predictions.csv")


# ===========================================================================
# FIG 2 — Triple ablation  (layout: 2 rows — A+C top, B full-width bottom)
# ===========================================================================
def make_fig2():
    print("Fig 2: triple ablation...")

    # ---- Load data ----
    with open(ROOT / "data" / "ablation_results_20260307.json") as f:
        abl = json.load(f)
    with open(ROOT / "data" / "spatial_ablation_20260307.json") as f:
        spatial = json.load(f)
    with open(ROOT / "data" / "temporal_ablation_20260307.json") as f:
        temporal = json.load(f)

    # ---- Panel A data ----
    grp_labels = {
        "only_spatial":  "Spatial\ncontagion",
        "only_infra":    "Infrastructure",
        "only_pop":      "Population",
        "only_hansen":   "Hansen\n(local)",
        "only_climate":  "Climate",
        "only_country":  "Country",
        "only_precip":   "Precipitation",
        "only_ntl":      "Night-time\nlights",
        "only_economy":  "Economy",
    }
    order = list(grp_labels.keys())
    single_groups = [r for r in abl["results"] if r["name"].startswith("only_")]
    single_sorted = sorted(
        [r for r in single_groups if r["name"] in grp_labels],
        key=lambda r: order.index(r["name"])
    )
    df_dim = pd.DataFrame([{
        "label": grp_labels[r["name"]],
        "auc":   r["val_auc_roc"],
        "prauc": r["val_pr_auc"],
        "n":     r["n_features"],
    } for r in single_sorted])

    # ---- Panel B data ----
    buf_experiments = [e for e in spatial if e.get("experiment") == "buffer_radius"]
    single_buf = [e for e in buf_experiments if len(e["radii"]) == 1]
    key_pairs = ["150m+500m", "150m+1500m", "all_radii"]
    pair_buf = [e for e in buf_experiments if e.get("label") in key_pairs]
    buf_show = single_buf + pair_buf

    def _buf_label(s):
        s = s.replace("only_", "").replace("all_radii", "All radii")
        s = s.replace("m+", " m + ").replace("m", " m").replace("  m", " m")
        return s.strip()

    df_buf = pd.DataFrame([{
        "label": _buf_label(e["label"]),
        "auc":   e["val_auc_roc"],
        "prauc": e["val_pr_auc"],
    } for e in buf_show])
    df_buf = df_buf.sort_values("prauc", ascending=True).reset_index(drop=True)

    # ---- Panel C data ----
    win_exp = [e for e in temporal if e.get("experiment") == "window_depth"]
    df_win = pd.DataFrame([{
        "window": e["window"],
        "auc":    e["val_auc_roc"],
        "prauc":  e["val_pr_auc"],
    } for e in win_exp]).sort_values("window")

    enc_exp = [e for e in temporal if e.get("experiment") == "encoding_type"]
    enc_labels = {
        "lags_only":      "Lags only",
        "lags_summaries": "Lags+summaries",
        "lags_deltas":    "Lags+deltas",
        "full":           "Full encoding",
    }
    df_enc = pd.DataFrame([{
        "label": enc_labels.get(e["encoding"], e["encoding"]),
        "auc":   e["val_auc_roc"],
        "prauc": e["val_pr_auc"],
        "n":     e["n_features"],
    } for e in enc_exp])

    # ---- Layout: 2 rows ----
    # Row 0: A (left) + C (right)
    # Row 1: B (full width)
    fig = plt.figure(figsize=(13, 8))
    gs = gridspec.GridSpec(
        2, 2,
        height_ratios=[1, 1.15],
        hspace=0.52,
        wspace=0.38,
    )

    # ---- Panel A ----
    ax_a = fig.add_subplot(gs[0, 0])
    x_a = np.arange(len(df_dim))
    ax_a.bar(x_a, df_dim["auc"], color=PALETTE["indigo"], alpha=0.85,
             width=0.5, label="AUC-ROC")
    ax_a.axhline(0.5, color=PALETTE["slate"], linewidth=0.9, linestyle=":")
    ax_a_twin = ax_a.twinx()
    ax_a_twin.plot(x_a, df_dim["prauc"], "o--", color=PALETTE["coral"],
                   linewidth=2, markersize=7, label="PR-AUC")
    ax_a_twin.spines["top"].set_visible(False)
    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(df_dim["label"], rotation=45, ha="right", fontsize=10)
    ax_a.set_ylim(0.45, 1.02)
    ax_a.set_ylabel("Val AUC-ROC")
    ax_a_twin.set_ylabel("Val PR-AUC", color=PALETTE["coral"])
    ax_a_twin.tick_params(axis="y", colors=PALETTE["coral"], labelsize=11)
    ax_a_twin.set_ylim(0, 0.12)
    ax_a.set_title("(a) Feature group contribution", loc="left",
                   fontsize=14, fontweight="bold")
    h1, l1 = ax_a.get_legend_handles_labels()
    h2, l2 = ax_a_twin.get_legend_handles_labels()
    ax_a.legend(h1 + h2, l1 + l2, fontsize=10, frameon=False, loc="lower right")

    # ---- Panel C (top right) ----
    ax_c = fig.add_subplot(gs[0, 1])
    # Scale PR-AUC × 100 for readability, label axis as ×10⁻²
    ax_c.plot(df_win["window"], df_win["prauc"] * 100, "o-",
              color=PALETTE["indigo"], linewidth=2.2, markersize=8,
              label="Window depth")
    ax_c.set_xlabel("Window depth (lag years)")
    ax_c.set_ylabel("Val PR-AUC (×10⁻²)")
    ax_c.set_xticks(df_win["window"])
    y_lo = df_win["prauc"].min() * 100 * 0.995
    y_hi = df_win["prauc"].max() * 100 * 1.005
    ax_c.set_ylim(y_lo, y_hi)
    ax_c.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_c.set_title("(c) Temporal window depth", loc="left",
                   fontsize=14, fontweight="bold")

    # Encoding types as amber squares, positioned at window=4 with slight jitter
    enc_x_map = {
        "Lags only": 3.7, "Lags+summaries": 3.85, "Lags+deltas": 4.0, "Full encoding": 4.15
    }
    for _, row in df_enc.iterrows():
        xpos = enc_x_map.get(row["label"], 4.0)
        ax_c.scatter(xpos, row["prauc"] * 100,
                     marker="s", s=80, color=PALETTE["amber"],
                     zorder=5, alpha=0.95)
    ax_c.scatter([], [], marker="s", s=80, color=PALETTE["amber"],
                 label="Encoding type")
    ax_c.legend(fontsize=10, frameon=False, loc="lower right")

    # ---- Panel B (full width, bottom row) ----
    ax_b = fig.add_subplot(gs[1, :])
    y_b = np.arange(len(df_buf))
    # Color: gradient teal, highlight "All radii" with a distinct color
    bar_colors = [
        PALETTE["indigo"] if "All" in row["label"] else PALETTE["teal"]
        for _, row in df_buf.iterrows()
    ]
    bars = ax_b.barh(y_b, df_buf["prauc"], color=bar_colors, alpha=0.85, height=0.55)
    ax_b.set_yticks(y_b)
    ax_b.set_yticklabels(df_buf["label"], fontsize=12)
    ax_b.set_xlabel("Val PR-AUC")
    ax_b.set_title("(b) Buffer radius combination", loc="left",
                   fontsize=14, fontweight="bold")
    # Annotate AUC values to the right of each bar
    max_val = df_buf["prauc"].max()
    ax_b.set_xlim(0, max_val * 1.30)
    for i, row in df_buf.iterrows():
        ax_b.text(
            row["prauc"] + max_val * 0.015, i,
            f"AUC = {row['auc']:.3f}",
            va="center", fontsize=11, color="#1E293B",
        )

    fig.savefig(PAPER_FIG / "fig2_ablation.pdf")
    fig.savefig(PAPER_FIG / "fig2_ablation.png", dpi=300)
    plt.close(fig)

    # Export data
    df_dim.to_csv(DATA_DIR / "fig2a_dimensional_ablation.csv", index=False)
    df_buf.to_csv(DATA_DIR / "fig2b_spatial_ablation.csv", index=False)
    df_win.to_csv(DATA_DIR / "fig2c_temporal_ablation.csv", index=False)
    df_enc.to_csv(DATA_DIR / "fig2c_encoding_types.csv", index=False)
    print("  -> fig2_ablation.pdf + data/fig2*.csv")


# ===========================================================================
# FIG 3 — Lift curve (top) + SHAP importance (bottom, full width)
# ===========================================================================
def make_fig3():
    print("Fig 3: lift curve + SHAP...")

    pred = pd.read_parquet(ROOT / "data" / "app" / "predictions_val.parquet")
    shap_df = pd.read_csv(ROOT / "data" / "shap_importance_20260307.csv", index_col=0)
    shap_df.columns = ["mean_abs_shap"]
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)

    # ---- Layout: 2 rows ----
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.35], hspace=0.48)

    # ---- Panel A: Concentration curve ----
    ax_a = fig.add_subplot(gs[0])
    pred_sorted = pred.sort_values("proba", ascending=False).reset_index(drop=True)
    n = len(pred_sorted)
    n_pos = int(pred_sorted["target"].sum())
    cum_pos = pred_sorted["target"].cumsum()
    frac_screened = np.arange(1, n + 1) / n
    recall = cum_pos / n_pos

    ax_a.plot([0, 1], [0, 1], "--", color=PALETTE["slate"],
              linewidth=1.4, label="Random baseline", zorder=1)
    ax_a.plot(frac_screened, recall, color=PALETTE["indigo"],
              linewidth=2.5, label="Model (val 2024)", zorder=2)

    # Key operating points — annotations clearly OFF the curve
    annotation_cfg = [
        # pct,  xytext offset (relative to point),  label anchor
        (0.01,  (0.10,  0.12)),
        (0.05,  (0.14,  0.60)),
        (0.10,  (0.22,  0.82)),
    ]
    for pct, (txt_x, txt_y) in annotation_cfg:
        idx = int(pct * n) - 1
        r = float(recall.iloc[idx])
        ax_a.scatter([pct], [r], s=60, color=PALETTE["amber"],
                     zorder=6, linewidths=0)
        ax_a.annotate(
            f"{r:.0%} at {pct:.0%}",
            xy=(pct, r),
            xytext=(txt_x, txt_y),
            arrowprops=dict(
                arrowstyle="->",
                color=PALETTE["slate"],
                lw=1.0,
                shrinkA=4,
                shrinkB=4,
            ),
            fontsize=12,
            fontweight="bold",
            color="#1E293B",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
        )

    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.set_xlabel("Fraction of area screened")
    ax_a.set_ylabel("Recall (fraction of deforestation captured)")
    ax_a.set_title("(a) Concentration curve", loc="left",
                   fontsize=14, fontweight="bold")
    ax_a.legend(frameon=False, fontsize=11, loc="lower right")

    # Export lift data (sampled)
    step = max(1, n // 2000)
    pd.DataFrame({
        "frac_screened": frac_screened[::step],
        "recall": recall.values[::step],
    }).to_csv(DATA_DIR / "fig3a_lift_curve.csv", index=False)

    # ---- Panel B: SHAP importance (full width) ----
    ax_b = fig.add_subplot(gs[1])
    top15 = shap_df.head(15).copy()

    label_map = {
        "defo_rate_1500m_wmean":   "Defo. rate 1.5 km (window mean)",
        "defo_rate_500m_wmean":    "Defo. rate 500 m (window mean)",
        "treecover2000":           "Tree cover 2000",
        "defo_rate_5000m_wmean":   "Defo. rate 5 km (window mean)",
        "defo_rate_5000m_Lag1":    "Defo. rate 5 km (lag 1)",
        "defo_rate_150m_wmean":    "Defo. rate 150 m (window mean)",
        "cum_deforested_Lag1":     "Cumul. deforested (lag 1)",
        "defo_rate_1500m_Lag1":    "Defo. rate 1.5 km (lag 1)",
        "et_d1_wanom_Lag3":        "ET anomaly Δ1 (lag 3)",
        "defo_rate_5000m_Lag2":    "Defo. rate 5 km (lag 2)",
        "et_d1_wtrend":            "ET anomaly trend Δ1",
        "pop_Lag4":                "Population (lag 4)",
        "defo_rate_5000m_Lag3":    "Defo. rate 5 km (lag 3)",
        "defo_rate_5000m_d1_Lag1": "Defo. rate 5 km Δ1 (lag 1)",
        "temperature_2m_Lag2":     "Temperature 2 m (lag 2)",
    }
    top15.index = [label_map.get(f, f) for f in top15.index]

    # Gradient coloring by SHAP magnitude (YlGnBu from ~0.2 to 1.0)
    vals = top15["mean_abs_shap"].values
    norm = Normalize(vmin=0, vmax=vals.max())
    cmap = plt.cm.YlGnBu
    bar_colors = [cmap(norm(v) * 0.75 + 0.25) for v in vals]

    y = np.arange(len(top15))
    ax_b.barh(y, vals, color=bar_colors, alpha=0.90, height=0.65)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(top15.index, fontsize=11.5)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Mean |SHAP value|")
    ax_b.set_title("(b) Feature importance (TreeSHAP)", loc="left",
                   fontsize=14, fontweight="bold")

    # Group annotations to the right of each bar
    group_map = {
        "Defo. rate":    ("Spatial",   PALETTE["indigo"]),
        "Cumul.":        ("Spatial",   PALETTE["indigo"]),
        "Tree cover":    ("Spatial",   PALETTE["indigo"]),
        "ET anomaly":    ("Climate",   PALETTE["teal"]),
        "Temperature":   ("Climate",   PALETTE["teal"]),
        "Population":    ("Demography", PALETTE["emerald"]),
    }
    max_shap = vals.max()
    ax_b.set_xlim(0, max_shap * 1.40)
    for i, (feat, val) in enumerate(zip(top15.index, vals)):
        grp_label, grp_color = ("Other", PALETTE["slate"])
        for key, (lbl, col) in group_map.items():
            if key in feat:
                grp_label, grp_color = lbl, col
                break
        ax_b.text(
            val + max_shap * 0.02, i,
            grp_label,
            va="center", fontsize=9.5,
            color=grp_color, fontweight="bold",
        )

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=max_shap))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax_b, fraction=0.02, pad=0.01)
    cb.set_label("Mean |SHAP value|", fontsize=11)
    cb.ax.tick_params(labelsize=10)

    top15.to_csv(DATA_DIR / "fig3b_shap_top15.csv")

    fig.savefig(PAPER_FIG / "fig3_lift_shap.pdf")
    fig.savefig(PAPER_FIG / "fig3_lift_shap.png", dpi=300)
    plt.close(fig)
    print("  -> fig3_lift_shap.pdf + data/fig3*.csv")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    print("All figures saved to paper/figures/")
