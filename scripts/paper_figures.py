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
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PAPER_FIG = ROOT / "paper" / "figures"
DATA_DIR = ROOT / "paper" / "figures" / "data"
PAPER_FIG.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style: colorblind-safe palette, English labels, ≥10pt fonts
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colorblind-safe palette (Wong 2011)
CB = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "black":  "#000000",
}


# ===========================================================================
# FIG 1 — Study area (left) + Predicted risk map (right)
# ===========================================================================
def make_fig1():
    print("Fig 1: study area + risk map...")

    pred = pd.read_parquet(ROOT / "data" / "app" / "predictions_val.parquet")
    # Keep columns needed
    pred = pred[["lon", "lat", "target", "proba"]].copy()

    # Load country boundaries
    world = gpd.read_file(ROOT / "data" / "boundaries" / "ne_110m_countries.gpkg")
    # Clip to study region with buffer
    lon_min, lon_max = pred["lon"].min() - 0.5, pred["lon"].max() + 0.5
    lat_min, lat_max = pred["lat"].min() - 0.5, pred["lat"].max() + 0.5

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    # ---- Panel A: sampling density heatmap ----
    ax = axes[0]
    world.clip([lon_min, lat_min, lon_max, lat_max]).plot(
        ax=ax, facecolor="#e8e8e8", edgecolor="#888888", linewidth=0.5
    )
    # Hexbin of all points (training distribution)
    hb = ax.hexbin(
        pred["lon"], pred["lat"],
        gridsize=40,
        cmap="Blues",
        mincnt=1,
        linewidths=0.1,
    )
    # Overlay true positives (deforested pixels)
    pos = pred[pred["target"] == 1]
    ax.scatter(
        pos["lon"], pos["lat"],
        s=2, c=CB["red"], alpha=0.6, linewidths=0, label="Deforested 2024",
        rasterized=True,
    )
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("(a) Validation sample distribution", loc="left", fontsize=11)
    ax.legend(markerscale=4, frameon=False, loc="upper left", fontsize=9)

    # ---- Panel B: predicted risk (subsample for speed) ----
    ax = axes[1]
    rng = np.random.default_rng(42)
    idx = rng.choice(len(pred), size=min(40000, len(pred)), replace=False)
    sub = pred.iloc[idx]

    world.clip([lon_min, lat_min, lon_max, lat_max]).plot(
        ax=ax, facecolor="#e8e8e8", edgecolor="#888888", linewidth=0.5
    )
    sc = ax.scatter(
        sub["lon"], sub["lat"],
        c=sub["proba"],
        cmap="YlOrRd",
        s=1.5,
        vmin=0, vmax=1,
        alpha=0.7,
        linewidths=0,
        rasterized=True,
    )
    plt.colorbar(sc, ax=ax, label="Predicted risk", fraction=0.04, pad=0.02)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("")
    ax.set_title("(b) Predicted deforestation risk (val 2024)", loc="left", fontsize=11)

    plt.tight_layout()
    fig.savefig(PAPER_FIG / "fig1_study_area.pdf")
    fig.savefig(PAPER_FIG / "fig1_study_area.png", dpi=300)
    plt.close(fig)

    # Export underlying data
    pred[["lon", "lat", "target", "proba"]].to_csv(
        DATA_DIR / "fig1_predictions.csv", index=False
    )
    print("  -> fig1_study_area.pdf + data/fig1_predictions.csv")


# ===========================================================================
# FIG 2 — Triple ablation (3 panels)
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

    # ---- Panel A: dimensional ablation (single groups only) ----
    single_groups = [r for r in abl["results"] if r["name"].startswith("only_")]
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

    # ---- Panel B: spatial ablation (single radii + pairs + all) ----
    # We show the single-radius results and key pairs
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

    # ---- Panel C: temporal ablation (window depth only) ----
    win_exp = [e for e in temporal if e.get("experiment") == "window_depth"]
    df_win = pd.DataFrame([{
        "window": e["window"],
        "auc":    e["val_auc_roc"],
        "prauc":  e["val_pr_auc"],
        "n":      e["n_features"],
    } for e in win_exp])

    # Also encoding type
    enc_exp = [e for e in temporal if e.get("experiment") == "encoding_type"]
    enc_labels = {"lags_only": "Lags only", "lags_summaries": "Lags+summaries",
                  "lags_deltas": "Lags+deltas", "full": "Full encoding"}
    df_enc = pd.DataFrame([{
        "label": enc_labels.get(e["encoding"], e["encoding"]),
        "auc":   e["val_auc_roc"],
        "prauc": e["val_pr_auc"],
        "n":     e["n_features"],
    } for e in enc_exp])

    # ---- Layout: 1 row × 3 panels ----
    fig = plt.figure(figsize=(14, 4.5))
    gs = gridspec.GridSpec(1, 3, wspace=0.35)

    # ---- Panel A ----
    ax_a = fig.add_subplot(gs[0])
    x_a = np.arange(len(df_dim))
    bars = ax_a.bar(x_a, df_dim["auc"], color=CB["blue"], alpha=0.8,
                    label="AUC-ROC", width=0.4)
    ax_b_twin = ax_a.twinx()
    ax_b_twin.plot(x_a, df_dim["prauc"], "o--", color=CB["orange"],
                   linewidth=1.5, markersize=6, label="PR-AUC")
    ax_a.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(df_dim["label"], rotation=45, ha="right", fontsize=9)
    ax_a.set_ylim(0.45, 1.00)
    ax_a.set_ylabel("Val AUC-ROC")
    ax_b_twin.set_ylabel("Val PR-AUC", color=CB["orange"])
    ax_b_twin.tick_params(axis="y", colors=CB["orange"])
    ax_b_twin.set_ylim(0, 0.10)
    ax_a.set_title("(a) Feature group contribution", loc="left", fontsize=11)
    # Combined legend
    h1, l1 = ax_a.get_legend_handles_labels()
    h2, l2 = ax_b_twin.get_legend_handles_labels()
    ax_a.legend(h1 + h2, l1 + l2, fontsize=9, frameon=False,
                loc="lower right")

    # ---- Panel B: spatial (horizontal bar chart) ----
    ax_b = fig.add_subplot(gs[1])
    y_b = np.arange(len(df_buf))
    ax_b.barh(y_b, df_buf["prauc"], color=CB["green"], alpha=0.8)
    ax_b.set_yticks(y_b)
    ax_b.set_yticklabels(df_buf["label"], fontsize=9)
    ax_b.set_xlabel("Val PR-AUC")
    ax_b.set_title("(b) Buffer radius combination", loc="left", fontsize=11)
    # Annotate AUC values
    for i, row in df_buf.iterrows():
        ax_b.text(row["prauc"] + 0.001, i, f'AUC={row["auc"]:.3f}',
                  va="center", fontsize=8, color="#333333")
    ax_b.set_xlim(0, 0.09)

    # ---- Panel C: temporal ----
    ax_c = fig.add_subplot(gs[2])
    ax_c.plot(df_win["window"], df_win["prauc"], "o-", color=CB["blue"],
              linewidth=2, markersize=7, label="PR-AUC (window depth)")
    ax_c.set_xlabel("Window depth (years of lags)")
    ax_c.set_ylabel("Val PR-AUC")
    ax_c.set_xticks(df_win["window"])
    ax_c.set_ylim(0.065, 0.085)
    ax_c.set_title("(c) Temporal window depth", loc="left", fontsize=11)

    # Overlay encoding type as scatter
    enc_x = {
        "Lags only": 1, "Lags+summaries": 1.5, "Lags+deltas": 2, "Full encoding": 4
    }
    for _, row in df_enc.iterrows():
        ax_c.scatter(enc_x.get(row["label"], 1), row["prauc"],
                     marker="s", s=60, color=CB["orange"],
                     zorder=5, alpha=0.9)
    ax_c.scatter([], [], marker="s", s=60, color=CB["orange"],
                 label="Encoding type (at 4-yr window)")
    ax_c.legend(fontsize=9, frameon=False, loc="lower right")

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
# FIG 3 — Lift (concentration) curve + SHAP importance
# ===========================================================================
def make_fig3():
    print("Fig 3: lift curve + SHAP...")

    pred = pd.read_parquet(ROOT / "data" / "app" / "predictions_val.parquet")
    shap_df = pd.read_csv(ROOT / "data" / "shap_importance_20260307.csv",
                          index_col=0)
    shap_df.columns = ["mean_abs_shap"]
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # ---- Panel A: Lift / concentration curve ----
    ax = axes[0]
    pred_sorted = pred.sort_values("proba", ascending=False).reset_index(drop=True)
    n = len(pred_sorted)
    n_pos = pred_sorted["target"].sum()
    cum_pos = pred_sorted["target"].cumsum()
    frac_screened = np.arange(1, n + 1) / n
    recall = cum_pos / n_pos

    # Random baseline
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.2, label="Random")

    # Model lift curve
    ax.plot(frac_screened, recall, color=CB["blue"], linewidth=2,
            label="Model (val 2024)")

    # Annotate key operating points
    for pct, label_txt in [(0.01, "1%"), (0.05, "5%"), (0.10, "10%")]:
        _ = label_txt  # suppress unused variable
        idx = int(pct * n) - 1
        r = recall.iloc[idx]
        offset_x = 0.04 if pct <= 0.01 else 0.03
        offset_y = 0.10 if pct <= 0.01 else -0.05
        ax.annotate(
            f"{r:.0%} at {pct:.0%}",
            xy=(pct, r), xytext=(pct + offset_x, r - offset_y),
            arrowprops=dict(arrowstyle="-", color="#666", lw=0.8),
            fontsize=9, color="#333",
        )
        ax.scatter([pct], [r], s=40, color=CB["orange"], zorder=5)

    ax.set_xlabel("Fraction of area screened")
    ax.set_ylabel("Recall (fraction of deforestation captured)")
    ax.set_title("(a) Concentration curve", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.0)

    # Export lift data (sampled)
    step = max(1, n // 2000)
    lift_export = pd.DataFrame({
        "frac_screened": frac_screened[::step],
        "recall": recall.values[::step],
    })
    lift_export.to_csv(DATA_DIR / "fig3a_lift_curve.csv", index=False)

    # ---- Panel B: Top-15 SHAP feature importance ----
    ax = axes[1]
    top15 = shap_df.head(15).copy()

    # Readable feature name mapping
    label_map = {
        "defo_rate_1500m_wmean":  "Defo. rate 1.5 km (window mean)",
        "defo_rate_500m_wmean":   "Defo. rate 500 m (window mean)",
        "treecover2000":          "Tree cover 2000",
        "defo_rate_5000m_wmean":  "Defo. rate 5 km (window mean)",
        "defo_rate_5000m_Lag1":   "Defo. rate 5 km (lag 1)",
        "defo_rate_150m_wmean":   "Defo. rate 150 m (window mean)",
        "cum_deforested_Lag1":    "Cumul. deforested (lag 1)",
        "defo_rate_1500m_Lag1":   "Defo. rate 1.5 km (lag 1)",
        "et_d1_wanom_Lag3":       "ET anomaly Δ1 (lag 3)",
        "defo_rate_5000m_Lag2":   "Defo. rate 5 km (lag 2)",
        "et_d1_wtrend":           "ET anomaly trend Δ1",
        "pop_Lag4":               "Population (lag 4)",
        "defo_rate_5000m_Lag3":   "Defo. rate 5 km (lag 3)",
        "defo_rate_5000m_d1_Lag1":"Defo. rate 5 km Δ1 (lag 1)",
        "temperature_2m_Lag2":    "Temperature 2 m (lag 2)",
    }
    top15.index = [label_map.get(f, f) for f in top15.index]

    colors = [CB["blue"] if "Defo" in f or "Cumul" in f or "Tree" in f
              else CB["orange"] if "ET" in f or "Temp" in f or "sm" in f or "precip" in f.lower()
              else CB["green"] if "Pop" in f or "dist" in f.lower()
              else CB["sky"]
              for f in top15.index]

    y = np.arange(len(top15))
    ax.barh(y, top15["mean_abs_shap"].values, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(top15.index, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("(b) Feature importance (SHAP)", loc="left", fontsize=11)
    ax.invert_yaxis()

    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CB["blue"], label="Spatial contagion / Hansen"),
        Patch(facecolor=CB["orange"], label="Climate / soil"),
        Patch(facecolor=CB["green"], label="Infrastructure / demography"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, frameon=False,
              loc="lower right")

    top15.to_csv(DATA_DIR / "fig3b_shap_top15.csv")

    plt.tight_layout()
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
