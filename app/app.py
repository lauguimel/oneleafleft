"""Streamlit app — OneLeafLeft: Deforestation Prediction in the Congo Basin.

Single-page scrollable narrative.

Run:
    cd /Users/guillaume/Documents/Recherche/Deforestation
    streamlit run app/app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DATA = Path(__file__).resolve().parent.parent / "data" / "app"
ABLATION_DIR = Path(__file__).resolve().parent.parent / "data"

# ── Human-readable feature names ────────────────────────────────────────────

FEATURE_LABELS = {
    # Spatial contagion
    "defo_rate_150m_wmean": "Nearby deforestation (150m avg)",
    "defo_rate_500m_wmean": "Nearby deforestation (500m avg)",
    "defo_rate_1500m_wmean": "Nearby deforestation (1.5km avg)",
    "defo_rate_5000m_wmean": "Nearby deforestation (5km avg)",
    "defo_rate_150m_Lag1": "Nearby deforestation 150m (last year)",
    "defo_rate_500m_Lag1": "Nearby deforestation 500m (last year)",
    "defo_rate_1500m_Lag1": "Nearby deforestation 1.5km (last year)",
    "defo_rate_5000m_Lag1": "Nearby deforestation 5km (last year)",
    "defo_rate_150m_Lag2": "Nearby deforestation 150m (2 years ago)",
    "defo_rate_500m_Lag2": "Nearby deforestation 500m (2 years ago)",
    "defo_rate_1500m_Lag2": "Nearby deforestation 1.5km (2 years ago)",
    "defo_rate_5000m_Lag2": "Nearby deforestation 5km (2 years ago)",
    "defo_rate_150m_wtrend": "Nearby deforestation 150m (trend)",
    "defo_rate_500m_wtrend": "Nearby deforestation 500m (trend)",
    "defo_rate_1500m_wtrend": "Nearby deforestation 1.5km (trend)",
    "defo_rate_5000m_wtrend": "Nearby deforestation 5km (trend)",
    # Hansen
    "treecover2000": "Initial tree cover (year 2000)",
    "cum_deforested_Lag1": "Cumulative deforestation (last year)",
    "cum_deforested_Lag2": "Cumulative deforestation (2 years ago)",
    "lossyear_Lag1": "Forest loss last year",
    "lossyear_Lag2": "Forest loss 2 years ago",
    # Infrastructure
    "dist_road_km": "Distance to nearest road (km)",
    "dist_settlement_km": "Distance to nearest village (km)",
    "in_protected": "Inside protected area",
    "dist_protected_km": "Distance to protected area (km)",
    "elevation": "Elevation (m)",
    "slope": "Terrain slope (°)",
    # Population
    "pop_Lag1": "Population density (last year)",
    "pop_Lag2": "Population density (2 years ago)",
    "pop_Lag3": "Population density (3 years ago)",
    "pop_Lag4": "Population density (4 years ago)",
    "pop_wmean": "Population (window avg)",
    "pop_wtrend": "Population growth trend",
    # Night lights
    "ntl_mean_Lag1": "Night light mean (last year)",
    "ntl_max_Lag1": "Night light max (last year)",
    "ntl_cv_Lag1": "Night light variability (last year)",
    # Climate
    "temperature_2m_Lag1": "Temperature (last year)",
    "temperature_2m_Lag2": "Temperature (2 years ago)",
    "et_d1_wanom_Lag3": "Evapotranspiration anomaly (3 years ago)",
    "et_d1_wtrend": "Evapotranspiration trend",
    "hot_days_Lag1": "Extreme heat days (last year)",
    "sm_surface_Lag1": "Soil moisture (last year)",
}

GROUP_LABELS = {
    "spatial (buffers)": "Neighbourhood deforestation",
    "hansen": "Forest cover history",
    "infra": "Infrastructure & terrain",
    "pop": "Population",
    "ntl": "Night light activity",
    "climate": "Climate",
    "precip": "Rainfall",
    "country": "Country",
    "other": "Other",
}


def humanize(feat: str) -> str:
    if feat in FEATURE_LABELS:
        return FEATURE_LABELS[feat]
    return feat.replace("_", " ").replace("Lag", "lag ").title()


def classify_feature(feat: str) -> str:
    if feat.startswith("defo_rate_"):
        return "spatial (buffers)"
    if feat in ("treecover2000", "cum_deforested_Lag1", "cum_deforested_Lag2",
                "lossyear_Lag1", "lossyear_Lag2") or feat.startswith("lossyear_"):
        return "hansen"
    if feat in ("dist_road_km", "dist_settlement_km", "in_protected",
                 "dist_protected_km", "elevation", "slope") or feat.startswith("iucn_") or feat.startswith("pa_"):
        return "infra"
    if feat.startswith("pop_"):
        return "pop"
    if feat.startswith("ntl_"):
        return "ntl"
    if any(feat.startswith(p) for p in ("temperature_", "dry_days_", "hot_days_",
                                         "et_", "sm_", "extreme_rain_")):
        return "climate"
    if feat.startswith("precip_"):
        return "precip"
    if feat.startswith("country_"):
        return "country"
    return "other"


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="OneLeafLeft — Deforestation Prediction",
    page_icon="\U0001F343",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; }
    [data-testid="stSidebar"] .block-container { padding-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Data loading (cached) ───────────────────────────────────────────────────


@st.cache_data
def load_predictions():
    return pd.read_parquet(APP_DATA / "predictions_val.parquet")


@st.cache_data
def load_model_info():
    with open(APP_DATA / "model_info.json") as f:
        return json.load(f)


@st.cache_data
def load_feature_importance():
    return pd.read_csv(APP_DATA / "feature_importance.csv")


@st.cache_data
def load_ablation(name):
    path = ABLATION_DIR / name
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ── Load data ────────────────────────────────────────────────────────────────

info = load_model_info()
pred = load_predictions()
y_true = pred["target"].values
y_proba = pred["proba"].values
threshold = info["threshold"]


# ── Sidebar (metrics only, no navigation) ────────────────────────────────────

st.sidebar.markdown(
    """
    # OneLeafLeft
    ### Deforestation prediction in the Congo Basin

    Machine learning model identifying forest areas
    at risk of deforestation in Central Africa.

    ---
    """
)

st.sidebar.markdown("#### Key metrics")
col_a, col_b = st.sidebar.columns(2)
col_a.metric("AUC-ROC", f"{info['val_auc_roc']:.1%}")
col_b.metric("Deforestation rate", f"{info['val_pos']/info['val_n']*100:.2f}%")

st.sidebar.markdown(
    f"""
    <small>
    {info['val_n']:,} locations analysed<br>
    {info['n_features']} environmental features<br>
    Congo Basin, Central Africa
    </small>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <small>
    <i>Guillaume Laumier & Marc Bouvier</i><br>
    Universite Grenoble Alpes / NitiDae<br><br>
    Data: Google Earth Engine, OSM, World Bank<br>
    Model: XGBoost + TreeSHAP
    </small>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# SECTION 1 — What we did
# ==========================================================================

st.markdown(
    """
    <div style="text-align: center; padding: 0; margin-top: -0.5rem;">
    <h1 style="margin: 0; font-size: 2.5em;">OneLeafLeft</h1>
    <p style="font-size: 1.2em; color: #555; margin-top: 0.2em;">
    Predicting where deforestation will strike next in the Congo Basin
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("")

# Key numbers
col1, col2, col3, col4 = st.columns(4)
col1.metric("Locations", f"{info['val_n']:,}")
col2.metric("Ranking accuracy (AUC)", f"{info['val_auc_roc']:.1%}")
col3.metric("Data sources", "33")
col4.metric("Inference time", "< 1 second")

st.markdown("---")

# Why + What
left, right = st.columns(2)

with left:
    st.markdown(
        """
        ### Why predict deforestation?

        The **Congo Basin** is the world's second-largest tropical rainforest,
        spanning **6 countries** across Central Africa. It is a critical carbon
        sink and biodiversity hotspot.

        Every year, thousands of hectares are lost — driven by agriculture,
        logging, roads, and population growth. But conservation resources
        are scarce.

        **If we can predict *where* deforestation will happen, we can act
        *before* the trees fall** — directing patrols, policy, and funding
        to the areas that need them most.
        """
    )

with right:
    st.markdown(
        """
        ### What does this model do?

        We analyse **250,000 forest locations** and estimate the
        **probability of deforestation in the next year** (2024).

        The model learns from 8 years of satellite data (2016–2023)
        and produces a **risk score** for each location.

        It does not detect deforestation after the fact —
        it **predicts it before it happens**.

        By screening just the **top 10%** highest-risk areas,
        we capture **91% of actual deforestation** —
        **9 times more efficiently** than random monitoring.
        """
    )

st.markdown("---")

# Data sources
st.markdown("### Data sources")
st.markdown("The model combines **33 variables** from satellite imagery and public databases:")

data_cols = st.columns(4)
with data_cols[0]:
    st.markdown(
        """
        **Satellite**
        - Forest cover & loss history
        - Neighbourhood deforestation
        - Night light intensity

        *Hansen GFC, VIIRS*
        """
    )
with data_cols[1]:
    st.markdown(
        """
        **Infrastructure**
        - Distance to roads
        - Distance to villages
        - Protected area status

        *OpenStreetMap, WDPA*
        """
    )
with data_cols[2]:
    st.markdown(
        """
        **Environment**
        - Elevation & slope
        - Temperature & rainfall
        - Dry season length

        *SRTM, ERA5, CHIRPS*
        """
    )
with data_cols[3]:
    st.markdown(
        """
        **Socio-economic**
        - Population density
        - Population growth
        - Governance indicators

        *WorldPop, World Bank*
        """
    )

st.markdown("---")

# How it works
st.markdown("### How it works")
steps = st.columns(5)
step_titles = ["1. Extract", "2. Encode", "3. Split", "4. Train", "5. Explain"]
step_descs = [
    "~140 features per location from Google Earth Engine",
    "Relative lags (*last year*, *2 years ago*) instead of absolute dates",
    "Train on 2016\u20132022, predict 2024 — no future leakage",
    "XGBoost gradient boosting — fast, accurate, interpretable",
    "SHAP values reveal *why* each location is at risk",
]
for col, title, desc in zip(steps, step_titles, step_descs):
    col.markdown(
        f"""
        <div style="text-align: center;">
        <b>{title}</b><br>
        <small style="color: #666;">{desc}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================================
# SECTION 2 — Results
# ==========================================================================

st.markdown("---")
st.markdown("## Results")

# ── Risk Map ──
st.markdown("### Deforestation risk map")
st.markdown(
    "Each dot represents a forest location. Darker dots have a higher "
    "probability of being deforested in 2024."
)

n_show = st.slider("Number of high-risk locations to display", 500, 10000, 3000, step=500)
df_plot = pred.nlargest(n_show, "proba").copy()
df_plot["risk_pct"] = (df_plot["proba"] * 100).round(1)

fig_map = px.scatter_mapbox(
    df_plot,
    lat="lat", lon="lon",
    color="risk_pct",
    color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c", "#8b0000"],
    range_color=[df_plot["risk_pct"].quantile(0.05), df_plot["risk_pct"].max()],
    hover_data={"risk_pct": ":.1f", "lat": ":.3f", "lon": ":.3f"},
    labels={"risk_pct": "Risk (%)"},
    zoom=6, height=550,
    mapbox_style="carto-positron",
    center={"lat": 5.1, "lon": 25.0},
)
fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig_map, use_container_width=True, config={"scrollZoom": True})
st.caption(
    f"Showing the **{n_show:,}** highest-risk locations. "
    f"Risk ranges from {df_plot['risk_pct'].min():.1f}% to {df_plot['risk_pct'].max():.1f}%. "
    f"Mouse wheel to zoom, drag to pan."
)

# ── Key callout ──
st.markdown("")
call_left, call_right = st.columns(2)
call_left.success(
    "**9x more efficient than random.** Screening just 10% of the area "
    "captures 91% of actual deforestation events."
)
call_right.info(
    "**Runs in under 1 second** on a standard laptop, no GPU required. "
    "Suitable for operational deployment in resource-constrained settings."
)

# ── Demo zone: high-resolution 2025 forecast ──
demo_path = APP_DATA / "demo_zone_predictions.parquet"
if demo_path.exists():
    st.markdown("### High-resolution forecast (2025)")
    st.markdown(
        "To demonstrate operational use, we generated a **dense prediction grid** "
        "(250 m spacing) over a ~30 km region near **Virunga, eastern DRC** — "
        "one of the most active deforestation frontiers in Central Africa."
    )

    col_overview, col_zoom = st.columns([1, 2])
    with col_overview:
        st.markdown(
            "**Overview**: the red rectangle shows the zoom area "
            "within the full study region."
        )
        fig_ov = px.scatter_mapbox(
            pred.sample(min(5000, len(pred)), random_state=42),
            lat="lat", lon="lon",
            color_discrete_sequence=["#94A3B8"],
            zoom=4, height=300,
            mapbox_style="carto-positron",
            center={"lat": 5.0, "lon": 25.0},
        )
        # Draw bbox rectangle
        fig_ov.add_trace(go.Scattermapbox(
            lon=[27.00, 27.30, 27.30, 27.00, 27.00],
            lat=[2.40, 2.40, 2.70, 2.70, 2.40],
            mode="lines",
            line=dict(color="#e74c3c", width=3),
            name="Zoom area",
        ))
        fig_ov.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
        st.plotly_chart(fig_ov, use_container_width=True)

    with col_zoom:
        demo = pd.read_parquet(demo_path)
        fig_demo = px.scatter_mapbox(
            demo,
            lat="lat", lon="lon",
            color="risk_pct",
            color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c", "#8b0000"],
            range_color=[0, min(demo["risk_pct"].quantile(0.99), 100)],
            hover_data={"risk_pct": ":.1f", "lat": ":.4f", "lon": ":.4f"},
            labels={"risk_pct": "Risk (%)"},
            zoom=10, height=450,
            mapbox_style="carto-positron",
            center={"lat": demo.lat.mean(), "lon": demo.lon.mean()},
        )
        fig_demo.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_demo, use_container_width=True, config={"scrollZoom": True})

    st.caption(
        f"**True forecast**: {len(demo):,} locations predicted for 2025. "
        "No 2025 deforestation data exists yet — this is a genuine forward prediction. "
        f"Mean risk: {demo['risk_pct'].mean():.1f}%, "
        f"Max risk: {demo['risk_pct'].max():.1f}%."
    )
    st.markdown("")

# ── Screening efficiency ──
st.markdown("### Screening efficiency")
st.markdown(
    "Ranking all locations by predicted risk and checking them from "
    "highest to lowest: how much deforestation do we capture?"
)

total_pos = y_true.sum()
order = np.argsort(-y_proba)
y_sorted = y_true[order]
cumsum = np.cumsum(y_sorted)
n = len(y_true)
pct_screened = np.arange(1, n + 1) / n * 100
pct_captured = cumsum / total_pos * 100

# Milestones
milestones = []
for target_pct in [50, 75, 90]:
    idx_target = np.searchsorted(pct_captured, target_pct)
    if idx_target < n:
        milestones.append({
            "target": target_pct,
            "screened": pct_screened[idx_target],
        })

cols = st.columns(len(milestones))
for col, m in zip(cols, milestones):
    col.metric(
        f"To catch {m['target']}% of deforestation",
        f"Screen {m['screened']:.1f}% of area",
        f"{100/m['screened']*m['target']/100:.0f}x more efficient than random",
    )

# Lift curve
step = max(1, n // 1000)
fig_lift = go.Figure()
fig_lift.add_trace(go.Scatter(
    x=pct_screened[::step], y=pct_captured[::step],
    mode="lines", name="Model",
    line=dict(color="#27ae60", width=3),
    fill="tozeroy", fillcolor="rgba(39,174,96,0.1)",
))
fig_lift.add_trace(go.Scatter(
    x=[0, 100], y=[0, 100], mode="lines",
    line=dict(dash="dash", color="gray", width=1), name="Random screening",
))
fig_lift.update_layout(
    xaxis_title="% of area screened (highest risk first)",
    yaxis_title="% of deforestation captured",
    height=400,
    legend=dict(x=0.6, y=0.3),
    margin=dict(t=10),
)
st.plotly_chart(fig_lift, use_container_width=True)

st.caption(
    "The green curve shows that screening a small fraction of the area "
    "(starting from highest risk) captures most of the deforestation. "
    "The dashed line shows random screening."
)

# Map: actual vs predicted
with st.expander("Model predictions vs reality"):
    st.markdown(
        """
        Locations where deforestation **actually happened** in 2024.
        **Red**: correctly predicted. **Blue**: missed by model.
        """
    )

    defo = pred[pred["target"] == 1].copy()
    defo["status"] = np.where(
        defo["proba"] >= threshold,
        "Correctly predicted",
        "Missed by model",
    )

    fig_vs = px.scatter_mapbox(
        defo,
        lat="lat", lon="lon",
        color="status",
        color_discrete_map={
            "Correctly predicted": "#e74c3c",
            "Missed by model": "#3498db",
        },
        hover_data={"proba": ":.3f"},
        labels={"proba": "Risk score"},
        zoom=6, height=500,
        mapbox_style="carto-positron",
        center={"lat": 5.1, "lon": 25.0},
    )
    fig_vs.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_vs, use_container_width=True, config={"scrollZoom": True})

    n_detected = (defo["status"] == "Correctly predicted").sum()
    n_missed = (defo["status"] == "Missed by model").sum()
    st.markdown(
        f"Out of **{len(defo):,}** deforested locations: "
        f"**{n_detected:,}** detected ({n_detected/len(defo)*100:.0f}%), "
        f"**{n_missed:,}** missed ({n_missed/len(defo)*100:.0f}%)."
    )


# ==========================================================================
# SECTION 3 — Why it's exceptional
# ==========================================================================

st.markdown("---")
st.markdown("## Key drivers")

imp = load_feature_importance()
imp["label"] = imp["feature"].apply(humanize)
imp["group"] = imp["feature"].apply(classify_feature)
imp["group_label"] = imp["group"].map(GROUP_LABELS).fillna(imp["group"])

# Group-level chart
st.markdown(
    "Which **types** of information matter most? "
    "Neighbourhood deforestation is by far the strongest predictor."
)

group_imp = imp.groupby("group_label")["mean_abs_shap"].sum().sort_values(ascending=True)

fig_group = go.Figure()
fig_group.add_trace(go.Bar(
    x=group_imp.values,
    y=group_imp.index,
    orientation="h",
    marker_color=["#27ae60" if v == group_imp.max() else "#95a5a6"
                   for v in group_imp.values],
))
fig_group.update_layout(
    xaxis_title="Total importance (mean |SHAP|)",
    height=350,
    margin=dict(l=0, r=0, t=5, b=0),
)
st.plotly_chart(fig_group, use_container_width=True)

# Top individual features
st.markdown("#### Top individual factors")

n_top = 15
top = imp.head(n_top).copy()

fig_feat = go.Figure()
fig_feat.add_trace(go.Bar(
    x=top["mean_abs_shap"].values[::-1],
    y=top["label"].values[::-1],
    orientation="h",
    marker_color="#2ecc71",
))
fig_feat.update_layout(
    xaxis_title="Average impact on prediction",
    height=max(350, n_top * 28),
    margin=dict(l=0, r=0, t=5, b=0),
)
st.plotly_chart(fig_feat, use_container_width=True)

st.markdown(
    """
    **Deforestation is contagious.** The strongest predictor is whether
    *nearby* forest was recently cleared. This contagion effect operates
    from 150 m to 5 km. Roads, settlements, and initial forest cover
    are the next most important factors.
    """
)

with st.expander("Full feature list"):
    display_df = imp[["label", "group_label", "mean_abs_shap"]].copy()
    display_df.columns = ["Factor", "Category", "Importance"]
    st.dataframe(
        display_df.style.format({"Importance": "{:.4f}"}),
        height=500,
        use_container_width=True,
    )

# ── Inspect a location ──
with st.expander("Inspect a specific location"):
    st.markdown("Select a high-risk location to see **why** the model thinks it's at risk.")

    top_risk = pred.nlargest(50, "proba")
    idx = st.selectbox(
        "Location",
        top_risk.index.tolist(),
        format_func=lambda i: (
            f"Lat {pred.loc[i, 'lat']:.3f}, Lon {pred.loc[i, 'lon']:.3f} "
            f"— Risk: {pred.loc[i, 'proba']*100:.1f}% "
            f"{'(actually deforested)' if pred.loc[i, 'target'] == 1 else ''}"
        ),
    )

    row = pred.loc[idx]
    drivers = []
    for k in range(1, 6):
        feat = row.get(f"shap_feat_{k}", "")
        val = row.get(f"shap_val_{k}", 0)
        if feat:
            drivers.append({
                "Factor": humanize(feat),
                "Impact": val,
                "Direction": "Increases risk" if val > 0 else "Decreases risk",
            })

    if drivers:
        st.markdown(f"**Risk score: {row['proba']*100:.1f}%** | "
                    f"{'Actually deforested' if row['target'] == 1 else 'Not deforested'}")
        st.markdown("**Top 5 factors driving this prediction:**")

        driver_df = pd.DataFrame(drivers)
        fig_d = px.bar(
            driver_df, x="Impact", y="Factor",
            orientation="h",
            color="Impact",
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            height=220,
        )
        fig_d.update_layout(
            margin=dict(l=0, r=0, t=5, b=0),
            yaxis=dict(autorange="reversed"),
            xaxis_title="Impact on risk (positive = increases risk)",
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_d, use_container_width=True)


# ==========================================================================
# SECTION 4 — Ablation experiments
# ==========================================================================

st.markdown("---")
st.markdown("## Ablation experiments")
st.markdown(
    "We systematically tested which information the model actually needs, "
    "removing groups of features to measure their contribution."
)

# Feature groups
st.markdown("#### Which data sources matter?")

fg_data = load_ablation("ablation_results_20260307.json")
if fg_data:
    results = fg_data["results"]
    indiv = [r for r in results if r["name"].startswith("only_")]
    indiv_df = pd.DataFrame(indiv)

    name_map = {
        "only_spatial": "Neighbourhood deforestation",
        "only_hansen": "Forest cover history",
        "only_infra": "Infrastructure & terrain",
        "only_pop": "Population",
        "only_ntl": "Night lights",
        "only_climate": "Climate",
        "only_precip": "Rainfall",
        "only_economy": "Economy",
        "only_country": "Country",
    }
    indiv_df["label"] = indiv_df["name"].map(name_map).fillna(indiv_df["name"])
    indiv_df = indiv_df.sort_values("val_auc_roc", ascending=True)

    fig_fg = go.Figure()
    fig_fg.add_trace(go.Bar(
        x=indiv_df["val_auc_roc"].values,
        y=indiv_df["label"].values,
        orientation="h",
        marker_color=["#27ae60" if v == indiv_df["val_auc_roc"].max() else "#3498db"
                       for v in indiv_df["val_auc_roc"].values],
        text=[f"{v:.1%}" for v in indiv_df["val_auc_roc"].values],
        textposition="outside",
    ))
    fig_fg.update_layout(
        xaxis_title="Prediction accuracy (AUC-ROC)",
        xaxis_range=[0.4, 1.0],
        height=400,
        margin=dict(l=0, r=50, t=5, b=0),
    )
    st.plotly_chart(fig_fg, use_container_width=True)

    st.markdown(
        "> Neighbourhood deforestation alone achieves **94.7%** accuracy. "
        "Adding infrastructure pushes it to **95.5%**. "
        "More data beyond that does not help."
    )

    with st.expander("Detailed results table"):
        full_df = pd.DataFrame(results)
        full_df["label_full"] = full_df["name"].map(name_map).fillna(full_df["name"])
        display = full_df[["label_full", "n_features", "val_auc_roc", "val_pr_auc"]].copy()
        display.columns = ["Scenario", "N features", "AUC-ROC", "PR-AUC"]
        st.dataframe(
            display.style.format({"AUC-ROC": "{:.4f}", "PR-AUC": "{:.4f}"}),
            use_container_width=True,
        )

# Temporal depth + Spatial scale side by side
col_temp, col_spat = st.columns(2)

with col_temp:
    st.markdown("#### How much history is needed?")

    temp_data = load_ablation("temporal_ablation_20260307.json")
    if temp_data:
        window_results = [r for r in temp_data if r.get("experiment") == "window_depth"]
        if window_results:
            tw_df = pd.DataFrame(window_results).sort_values("window")

            fig_tw = go.Figure()
            fig_tw.add_trace(go.Bar(
                x=[f"{w} year{'s' if w > 1 else ''}" for w in tw_df["window"]],
                y=tw_df["val_auc_roc"],
                marker_color="#3498db",
                text=[f"{v:.1%}" for v in tw_df["val_auc_roc"]],
                textposition="outside",
            ))
            fig_tw.update_layout(
                yaxis_title="Accuracy (AUC-ROC)",
                yaxis_range=[0.9, 0.97],
                height=300,
                margin=dict(t=5),
            )
            st.plotly_chart(fig_tw, use_container_width=True)

            st.markdown(
                "**1 year of history is enough.** "
                "Deforestation contagion is nearly instantaneous — "
                "what happened last year matters, older data barely helps."
            )

with col_spat:
    st.markdown("#### At what distance does deforestation spread?")

    spat_data = load_ablation("spatial_ablation_20260307.json")
    if spat_data:
        buf_results = [r for r in spat_data if r.get("experiment") == "buffer_radius"]
        indiv_buf = [r for r in buf_results if "+" not in r.get("label", "")]
        if indiv_buf:
            sb_df = pd.DataFrame(indiv_buf)
            radius_map = {
                "only_150m": "150 m",
                "only_500m": "500 m",
                "only_1500m": "1.5 km",
                "only_5000m": "5 km",
            }
            sb_df["label_clean"] = sb_df["label"].map(radius_map).fillna(sb_df["label"])

            fig_sb = go.Figure()
            fig_sb.add_trace(go.Bar(
                x=sb_df["label_clean"],
                y=sb_df["val_pr_auc"],
                marker_color="#e67e22",
                text=[f"{v:.3f}" for v in sb_df["val_pr_auc"]],
                textposition="outside",
            ))
            fig_sb.update_layout(
                yaxis_title="Detection precision (PR-AUC)",
                height=300,
                margin=dict(t=5),
            )
            st.plotly_chart(fig_sb, use_container_width=True)

            st.markdown(
                "**150 m radius gives the best precision.** "
                "Combining 150 m with 1.5 km captures both "
                "micro-local and landscape-level patterns."
            )


# ── Key findings summary ──
st.markdown("---")
st.markdown("### Key findings")
f1, f2 = st.columns(2)
with f1:
    st.success(
        "**Deforestation is contagious** — the strongest predictor is whether "
        "*nearby* forest was recently deforested. This effect operates "
        "from 150 m to 5 km."
    )
    st.info(
        "**Roads and settlements matter** — proximity to human infrastructure "
        "is the second most important factor after spatial contagion."
    )
with f2:
    st.warning(
        "**1 year of history is enough** — looking back further than 1 year "
        "barely improves predictions. Contagion is nearly instantaneous."
    )
    st.error(
        "**Climate and economics add little** — while relevant for understanding "
        "deforestation, they don't improve *prediction* beyond spatial contagion."
    )


# ==========================================================================
# SECTION 5 — Technical details
# ==========================================================================

st.markdown("---")

with st.expander("Technical details"):
    from sklearn.metrics import roc_curve, precision_recall_curve

    y_pred = (y_proba >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()

    st.markdown("#### Threshold-based metrics")
    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("Detected", f"{tp}/{tp+fn}", f"{tp/(tp+fn)*100:.0f}% recall")
    tc2.metric("False alarms", f"{fp:,}",
                f"{fp/(tp+fp)*100:.0f}% of alerts" if tp + fp > 0 else "N/A",
                delta_color="inverse")
    tc3.metric("Threshold", f"{threshold:.3f}",
                "Optimised on training data")
    st.caption(
        "With only 0.36% positive rate, even a good model produces many false alarms "
        "at any fixed threshold. The screening efficiency above is more meaningful."
    )

    st.markdown("#### Diagnostic curves")
    col_a, col_b = st.columns(2)

    with col_a:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                      line=dict(color="#2980b9", width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                      line=dict(dash="dash", color="gray")))
        fig_roc.update_layout(
            title=f"ROC Curve (AUC = {info['val_auc_roc']:.4f})",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=350, showlegend=False, margin=dict(t=40),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_b:
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                     line=dict(color="#e67e22", width=2)))
        baseline = y_true.mean()
        fig_pr.add_trace(go.Scatter(x=[0, 1], y=[baseline, baseline], mode="lines",
                                     line=dict(dash="dash", color="gray")))
        fig_pr.update_layout(
            title=f"Precision-Recall Curve (PR-AUC = {info['val_pr_auc']:.4f})",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=350, showlegend=False, margin=dict(t=40),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # Probability distribution
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=y_proba[y_true == 0], name="Forest preserved",
        nbinsx=100, opacity=0.7, marker_color="#27ae60",
    ))
    fig_hist.add_trace(go.Histogram(
        x=y_proba[y_true == 1], name="Actually deforested",
        nbinsx=50, opacity=0.7, marker_color="#e74c3c",
    ))
    fig_hist.update_layout(
        barmode="overlay",
        xaxis_title="Predicted risk score",
        yaxis_title="Number of locations (log scale)",
        yaxis_type="log", height=300,
        margin=dict(t=10),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown(f"**Confusion matrix** at threshold {threshold:.2f}:")
    cm_df = pd.DataFrame(
        [[f"{tn:,}", f"{fp:,}"], [f"{fn:,}", f"{tp:,}"]],
        index=["Actually safe", "Actually deforested"],
        columns=["Predicted safe", "Predicted at risk"],
    )
    st.dataframe(cm_df, use_container_width=False)

with st.expander("How to read the results (rare-event context)"):
    st.markdown(
        """
        Deforestation is a **very rare event** — only **0.36%** of locations are
        deforested in any given year. This has important consequences:

        - **The model ranks locations by risk**, it doesn't give a simple yes/no.
          A location with a 10% risk score is *not* certain to be deforested,
          but it is **28x more likely** than an average location.

        - **Traditional metrics can be misleading.** Even a perfect model would
          produce many "false alarms" because there are so few actual events
          compared to the vast forest.

        - **The right way to evaluate**: if we screen areas from highest to lowest risk,
          how quickly do we find the deforestation? Our model captures **91% of
          deforestation by screening just 10%** of the area.

        - **Think of it like medical screening**: you wouldn't expect a cancer
          screening to be perfect, but you'd want it to prioritise the right patients
          for follow-up. This model does the same for forests.
        """
    )


# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888; font-size: 0.85em;">
    <b>OneLeafLeft</b> — Deforestation prediction in the Congo Basin<br>
    XGBoost model trained on 730K observations | 250K locations | 2016–2024 data<br>
    <i>Guillaume Laumier & Marc Bouvier</i> | UGA / NitiDae
    </div>
    """,
    unsafe_allow_html=True,
)
