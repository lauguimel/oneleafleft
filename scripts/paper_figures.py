"""
paper_figures.py — Generate publication-quality figures for ERL manuscript.

Produces:
  paper/figures/fig1_study_area.{pdf,png}   + paper/figures/data/fig1_*.csv
  paper/figures/fig2_ablation.{pdf,png}     + paper/figures/data/fig2_*.csv
  paper/figures/fig3_lift_shap.{pdf,png}    + paper/figures/data/fig3_*.csv

Run: conda run -n deforest python scripts/paper_figures.py

Color system (coherent across all figures):
  BLUE_DARK  #1E40AF  — highlights / best value
  BLUE_MID   #2563EB  — primary bars / lines
  CORAL      #F43F5E  — deforested events / PR-AUC accent
  SLATE      #64748B  — neutral / baselines
  CMAP       Blues    — sequential gradient in Fig 2C and Fig 3B
"""

import json
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
DATA_DIR  = ROOT / "paper" / "figures" / "data"
PAPER_FIG.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style — coherent palette, readable fonts
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         13,
    "axes.labelsize":    13,
    "axes.titlesize":    14,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   12,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Coherent color system used in ALL three figures
BLUE_DARK = "#1E40AF"   # deepest blue — "best" bar / highlight
BLUE_MID  = "#2563EB"   # primary blue — main lines and bars
CORAL     = "#F43F5E"   # coral/rose  — deforestation events, PR-AUC
SLATE     = "#64748B"   # grey-slate  — baselines, neutral
CMAP      = plt.cm.Blues  # sequential gradient — Fig 2C spatial + Fig 3B SHAP


def _blues(values, lo=0.30, hi=0.90):
    """Map an array of values to Blues colormap, avoiding the very pale end."""
    norm = Normalize(vmin=min(values), vmax=max(values))
    return [CMAP(norm(v) * (hi - lo) + lo) for v in values]


# ===========================================================================
# FIG 1 — Study area scatter + Predicted risk map
# ===========================================================================
def make_fig1():
    print("Fig 1: study area + risk map...")

    pred = pd.read_parquet(ROOT / "data" / "app" / "predictions_val.parquet")
    pred = pred[["lon", "lat", "target", "proba"]].copy()

    world = gpd.read_file(ROOT / "data" / "boundaries" / "ne_110m_countries.gpkg")
    lon_min = pred["lon"].min() - 0.5
    lon_max = pred["lon"].max() + 0.5
    lat_min = pred["lat"].min() - 0.5
    lat_max = pred["lat"].max() + 0.5

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # ---- Panel A: scatter — grey background, coral deforested ----
    ax = axes[0]
    world.clip([lon_min, lat_min, lon_max, lat_max]).plot(
        ax=ax, facecolor="#E2E8F0", edgecolor="#94A3B8", linewidth=0.6
    )
    rng = np.random.default_rng(42)
    neg = pred[pred["target"] == 0]
    pos = pred[pred["target"] == 1]
    neg_sub = neg.iloc[rng.choice(len(neg), size=min(60_000, len(neg)), replace=False)]

    ax.scatter(neg_sub["lon"], neg_sub["lat"],
               s=10, c="#2d6a4f", alpha=0.20, linewidths=0,
               rasterized=True, label="No loss (sample)")
    ax.scatter(pos["lon"], pos["lat"],
               s=10, c=CORAL, alpha=0.85, linewidths=0,
               rasterized=True, label="Deforested 2024", zorder=5)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("(a) Validation sample (2024)", loc="left",
                 fontsize=14, fontweight="bold")
    leg = ax.legend(markerscale=2.5, frameon=True, fontsize=12,
                    loc="upper left", handletextpad=0.4,
                    facecolor="white", edgecolor="#CBD5E1", framealpha=0.92)

    # ---- Panel B: predicted risk map ----
    ax = axes[1]
    idx_b = rng.choice(len(pred), size=min(50_000, len(pred)), replace=False)
    sub = pred.iloc[idx_b].sort_values("proba")

    world.clip([lon_min, lat_min, lon_max, lat_max]).plot(
        ax=ax, facecolor="#E2E8F0", edgecolor="#94A3B8", linewidth=0.6
    )
    sc = ax.scatter(sub["lon"], sub["lat"],
                    c=sub["proba"], cmap="YlOrRd",
                    s=2, vmin=0, vmax=1, alpha=0.75,
                    linewidths=0, rasterized=True)
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Predicted risk", fontsize=12)
    cb.ax.tick_params(labelsize=10)

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
        DATA_DIR / "fig1_predictions.csv", index=False)
    print("  -> fig1_study_area.pdf")


# ===========================================================================
# FIG 2 — Triple ablation
# Layout: Row 0 = A (left) + B (right)   — bar chart + temporal line
#         Row 1 = C (centered, 60% width) — spatial bars
# ===========================================================================
def make_fig2():
    print("Fig 2: triple ablation...")

    with open(ROOT / "data" / "ablation_results_20260307.json") as f:
        abl = json.load(f)
    with open(ROOT / "data" / "spatial_ablation_20260307.json") as f:
        spatial = json.load(f)
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
    key_pairs  = ["150m+500m", "150m+1500m", "all_radii"]
    pair_buf   = [e for e in buf_experiments if e.get("label") in key_pairs]
    buf_show   = single_buf + pair_buf

    def _buf_label(s):
        s = s.replace("only_", "").replace("all_radii", "All radii")
        s = s.replace("m+", " m + ").replace("m", " m").replace("  m", " m")
        return s.strip()

    df_buf = pd.DataFrame([{
        "label": _buf_label(e["label"]),
        "auc":   e["val_auc_roc"],
        "prauc": e["val_pr_auc"],
    } for e in buf_show]).sort_values("prauc", ascending=True).reset_index(drop=True)

    # ---- Layout: 2 panels side by side ----
    fig = plt.figure(figsize=(14, 5))
    ax_a = fig.add_axes([0.06, 0.18, 0.36, 0.72])   # Panel A (left)
    ax_b = fig.add_axes([0.56, 0.18, 0.38, 0.72])   # Panel B (right)

    # ---- Panel A (dimensional ablation) ----
    x_a = np.arange(len(df_dim))
    ax_a.bar(x_a, df_dim["auc"], color=BLUE_MID, alpha=0.85, width=0.5)
    ax_a.axhline(0.5, color=SLATE, linewidth=0.9, linestyle=":")
    ax_a_r = ax_a.twinx()
    ax_a_r.plot(x_a, df_dim["prauc"], "o--", color=CORAL,
                linewidth=2, markersize=7)
    ax_a_r.spines["top"].set_visible(False)
    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(df_dim["label"], rotation=45, ha="right", fontsize=10)
    ax_a.set_ylim(0.45, 1.02)
    ax_a.set_ylabel("Val AUC-ROC")
    ax_a_r.set_ylabel("Val PR-AUC", color=CORAL)
    ax_a_r.tick_params(axis="y", colors=CORAL, labelsize=11)
    ax_a_r.set_ylim(0, 0.12)
    ax_a.set_title("(a) Feature group contribution", loc="left",
                   fontsize=14, fontweight="bold")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch as MPatch
    leg_handles = [
        MPatch(facecolor=BLUE_MID, label="AUC-ROC"),
        Line2D([0], [0], color=CORAL, marker="o", linewidth=2,
               markersize=7, label="PR-AUC"),
    ]
    ax_a.legend(handles=leg_handles, fontsize=11, frameon=False,
                loc="upper right")

    # ---- Panel B (spatial horizontal bars) ----
    y_b = np.arange(len(df_buf))
    colors_b = _blues(df_buf["prauc"].values)
    ax_b.barh(y_b, df_buf["prauc"], color=colors_b, alpha=0.90, height=0.60)
    ax_b.set_yticks(y_b)
    ax_b.set_yticklabels(df_buf["label"], fontsize=12)
    ax_b.set_xlabel("Val PR-AUC")
    ax_b.set_title("(b) Buffer radius combination", loc="left",
                   fontsize=14, fontweight="bold")
    max_b = df_buf["prauc"].max()
    ax_b.set_xlim(0, max_b * 1.32)
    for i, row in df_buf.iterrows():
        ax_b.text(row["prauc"] + max_b * 0.016, i,
                  f"AUC = {row['auc']:.3f}",
                  va="center", fontsize=11, color="#1E293B")

    fig.savefig(PAPER_FIG / "fig2_ablation.pdf")
    fig.savefig(PAPER_FIG / "fig2_ablation.png", dpi=300)
    plt.close(fig)

    df_dim.to_csv(DATA_DIR / "fig2a_dimensional_ablation.csv", index=False)
    df_buf.to_csv(DATA_DIR / "fig2b_spatial_ablation.csv",     index=False)
    print("  -> fig2_ablation.pdf")


# ===========================================================================
# FIG 3 — Lift curve (top) + SHAP importance (bottom, full width)
# ===========================================================================
def make_fig3():
    print("Fig 3: lift curve + SHAP...")

    pred    = pd.read_parquet(ROOT / "data" / "app" / "predictions_val.parquet")
    shap_df = pd.read_csv(ROOT / "data" / "shap_importance_20260307.csv", index_col=0)
    shap_df.columns = ["mean_abs_shap"]
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)

    fig = plt.figure(figsize=(9, 10))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 1.4], hspace=0.50)

    # ---- Panel A: Concentration curve — SQUARE axes ----
    ax_a = fig.add_subplot(gs[0])
    pred_sorted = pred.sort_values("proba", ascending=False).reset_index(drop=True)
    n     = len(pred_sorted)
    n_pos = int(pred_sorted["target"].sum())
    cum_pos      = pred_sorted["target"].cumsum()
    frac_screened = np.arange(1, n + 1) / n
    recall        = cum_pos / n_pos

    ax_a.plot([0, 1], [0, 1], "--", color=SLATE, linewidth=1.4,
              label="Random baseline", zorder=1)
    ax_a.plot(frac_screened, recall, color=BLUE_MID, linewidth=2.5,
              label="Model (val 2024)", zorder=2)

    # Annotations clearly OFF the curve
    annotation_cfg = [
        #  pct    xytext
        (0.01,  (0.14, 0.22)),
        (0.05,  (0.22, 0.58)),
        (0.10,  (0.38, 0.82)),
    ]
    for pct, (tx, ty) in annotation_cfg:
        idx = int(pct * n) - 1
        r   = float(recall.iloc[idx])
        ax_a.scatter([pct], [r], s=70, color=CORAL, zorder=6, linewidths=0)
        ax_a.annotate(
            f"{r:.0%} at {pct:.0%}",
            xy=(pct, r), xytext=(tx, ty),
            arrowprops=dict(arrowstyle="->", color=SLATE,
                            lw=1.0, shrinkA=5, shrinkB=5),
            fontsize=12, fontweight="bold", color="#1E293B",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#E2E8F0", alpha=0.9),
        )

    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.set_xlabel("Fraction of area screened")
    ax_a.set_ylabel("Deforestation captured")
    ax_a.set_title("(a) Concentration curve", loc="left",
                   fontsize=14, fontweight="bold")
    ax_a.legend(frameon=False, fontsize=11, loc="lower right")

    step = max(1, n // 2000)
    pd.DataFrame({
        "frac_screened": frac_screened[::step],
        "recall":        recall.values[::step],
    }).to_csv(DATA_DIR / "fig3a_lift_curve.csv", index=False)

    # ---- Panel B: SHAP — Blues gradient + group labels ----
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

    vals        = top15["mean_abs_shap"].values
    bar_colors  = _blues(vals)    # same Blues gradient as Fig 2B

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
        "Defo. rate":  ("Spatial",     BLUE_DARK),
        "Cumul.":      ("Spatial",     BLUE_DARK),
        "Tree cover":  ("Spatial",     BLUE_DARK),
        "ET anomaly":  ("Climate",     SLATE),
        "Temperature": ("Climate",     SLATE),
        "Population":  ("Demography",  SLATE),
    }
    max_shap = vals.max()
    ax_b.set_xlim(0, max_shap * 1.45)
    for i, (feat, val) in enumerate(zip(top15.index, vals)):
        grp_label, grp_color = "Other", SLATE
        for key, (lbl, col) in group_map.items():
            if key in feat:
                grp_label, grp_color = lbl, col
                break
        ax_b.text(val + max_shap * 0.025, i,
                  grp_label, va="center",
                  fontsize=9.5, color=grp_color, fontweight="semibold")

    # Colorbar on the right
    sm = ScalarMappable(cmap=CMAP, norm=Normalize(vmin=0, vmax=max_shap))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax_b, fraction=0.025, pad=0.01)
    cb.set_label("Mean |SHAP value|", fontsize=11)
    cb.ax.tick_params(labelsize=10)

    top15.to_csv(DATA_DIR / "fig3b_shap_top15.csv")

    fig.savefig(PAPER_FIG / "fig3_lift_shap.pdf")
    fig.savefig(PAPER_FIG / "fig3_lift_shap.png", dpi=300)
    plt.close(fig)
    print("  -> fig3_lift_shap.pdf")


# ===========================================================================
# FIG 4 — Demo zone: high-resolution 2025 forecast (optional)
# ===========================================================================
def make_fig4():
    demo_path = ROOT / "data" / "app" / "demo_zone_predictions.parquet"
    if not demo_path.exists():
        print("Fig 4: SKIPPED (demo_zone_predictions.parquet not found)")
        return

    print("Fig 4: demo zone forecast...")

    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    demo = pd.read_parquet(demo_path)

    # Bbox of demo zone
    bbox = [27.00, 27.30, 2.40, 2.70]  # lon_min, lon_max, lat_min, lat_max
    VIOLET = "#9b59b6"
    FOREST = "#2d6a4f"

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Common grid setup
    spacing = 0.0025
    lons = np.arange(bbox[0], bbox[1] + spacing / 2, spacing)
    lats = np.arange(bbox[2], bbox[3] + spacing / 2, spacing)
    lon_idx = np.round((demo["lon"].values - bbox[0]) / spacing).astype(int)
    lat_idx = np.round((demo["lat"].values - bbox[2]) / spacing).astype(int)
    valid = (lon_idx >= 0) & (lon_idx < len(lons)) & (lat_idx >= 0) & (lat_idx < len(lats))

    # ---- Panel A: cumulative deforestation by 2024 ----
    ax = axes[0]
    defo_grid = np.full((len(lats), len(lons)), np.nan)
    defo_grid[lat_idx[valid], lon_idx[valid]] = demo["deforested"].values[valid].astype(float)

    cmap_defo = ListedColormap([FOREST, VIOLET])
    ax.pcolormesh(lons, lats, defo_grid, cmap=cmap_defo,
                  vmin=0, vmax=1, rasterized=True, shading="nearest")
    ax.legend(handles=[
        Patch(facecolor=FOREST, label="Forest"),
        Patch(facecolor=VIOLET, label="Deforested"),
    ], loc="lower right", fontsize=9, framealpha=0.9)

    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("(a) Cumulative deforestation by 2024", loc="left",
                 fontsize=14, fontweight="bold")

    # ---- Panel B: 2025 forecast (risk gradient + violet for already deforested) ----
    ax = axes[1]
    is_defo = demo["deforested"].values

    # Layer 1: risk for forested pixels
    risk_grid = np.full((len(lats), len(lons)), np.nan)
    mask_forest = valid & ~is_defo
    risk_grid[lat_idx[mask_forest], lon_idx[mask_forest]] = \
        demo["proba"].values[mask_forest]
    pcm = ax.pcolormesh(lons, lats, risk_grid, cmap="RdYlGn_r",
                        vmin=0, vmax=np.nanquantile(risk_grid, 0.99),
                        rasterized=True, shading="nearest")

    # Layer 2: already deforested in violet
    defo_overlay = np.full((len(lats), len(lons)), np.nan)
    mask_defo = valid & is_defo
    defo_overlay[lat_idx[mask_defo], lon_idx[mask_defo]] = 1.0
    cmap_violet = ListedColormap([VIOLET])
    ax.pcolormesh(lons, lats, defo_overlay, cmap=cmap_violet,
                  vmin=0.5, vmax=1.5, rasterized=True, shading="nearest")

    cb = plt.colorbar(pcm, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Predicted risk (2025)", fontsize=12)
    cb.ax.tick_params(labelsize=10)

    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("(b) 2025 deforestation forecast", loc="left",
                 fontsize=14, fontweight="bold")

    plt.tight_layout(w_pad=2.5)
    fig.savefig(PAPER_FIG / "fig4_demo_zone.pdf")
    fig.savefig(PAPER_FIG / "fig4_demo_zone.png", dpi=300)
    plt.close(fig)

    demo[["lon", "lat", "proba"]].to_csv(
        DATA_DIR / "fig4_demo_zone.csv", index=False)
    print("  -> fig4_demo_zone.pdf")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    print("All figures saved to paper/figures/")
