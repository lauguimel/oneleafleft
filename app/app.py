"""Streamlit app — OneLeafLeft: Deforestation Prediction in the Congo Basin.

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
    "treecover2000": "Initial forest cover (year 2000)",
    "cum_deforested_Lag1": "Already deforested (cumulative)",
    "cum_deforested_Lag2": "Already deforested (2 years ago)",
    "cum_deforested_wmean": "Already deforested (avg)",
    # Infrastructure
    "dist_road_km": "Distance to nearest road (km)",
    "dist_settlement_km": "Distance to nearest village (km)",
    "in_protected": "Inside a protected area",
    "dist_protected_km": "Distance to protected area (km)",
    "elevation": "Elevation (m)",
    "slope": "Terrain slope (degrees)",
    # Population
    "pop_Lag1": "Population density (last year)",
    "pop_wmean": "Population density (avg)",
    "pop_wtrend": "Population growth (trend)",
    # Night lights
    "ntl_mean_Lag1": "Night light intensity (last year)",
    "ntl_mean_wmean": "Night light intensity (avg)",
    # Climate
    "temperature_2m_Lag1": "Temperature (last year)",
    "dry_days_Lag1": "Dry days (last year)",
    "precip_total_Lag1": "Total rainfall (last year)",
    # Country
    "country_COD": "Country: DR Congo",
    "country_CMR": "Country: Cameroon",
    "country_GAB": "Country: Gabon",
    "country_COG": "Country: Congo",
    "country_GNQ": "Country: Equatorial Guinea",
    "country_CAF": "Country: Central African Republic",
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
    """Return human-readable label for a feature name."""
    if feat in FEATURE_LABELS:
        return FEATURE_LABELS[feat]
    # Auto-generate for unregistered features
    s = feat.replace("_", " ").replace("wmean", "(avg)").replace("wtrend", "(trend)")
    s = s.replace("Lag1", "(last year)").replace("Lag2", "(2 years ago)")
    s = s.replace("defo rate", "deforestation").replace("dist ", "distance to ")
    return s.capitalize()


def classify_feature(feat: str) -> str:
    if feat.startswith("defo_rate_"):
        return "spatial (buffers)"
    if feat.startswith("cum_deforested") or feat == "treecover2000":
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
    page_icon="\U0001F343",  # leaf emoji
    layout="wide",
)

# Remove default top padding
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


# ── Sidebar ──────────────────────────────────────────────────────────────────

info = load_model_info()

st.sidebar.markdown(
    """
    # \U0001F343 OneLeafLeft
    ### Predicting deforestation in the Congo Basin

    This tool uses machine learning to identify forest areas
    at risk of deforestation in Central Africa.

    The model analyses **neighbourhood deforestation patterns**,
    **road proximity**, **population density** and other factors
    to estimate where deforestation is most likely to occur next.

    ---
    """
)

# Key numbers
st.sidebar.markdown("#### Model performance")
col_a, col_b = st.sidebar.columns(2)
col_a.metric("Accuracy (AUC)", f"{info['val_auc_roc']:.1%}")
col_b.metric("Deforestation rate", f"{info['val_pos']/info['val_n']*100:.2f}%")

st.sidebar.markdown(
    f"""
    <small>
    \U0001F4CA {info['val_n']:,} locations analysed<br>
    \U0001F333 {info['n_features']} environmental factors<br>
    \U0001F30D Congo Basin, Central Africa
    </small>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

PAGES = ["\U0001F30D About", "\U0001F5FA\uFE0F Risk Map", "\U0001F4CA Performance", "\U0001F50D Key Drivers", "\U0001F9EA Experiments"]

if "page" not in st.session_state:
    st.session_state["page"] = PAGES[0]

page = st.sidebar.radio(
    "Navigate",
    PAGES,
    index=PAGES.index(st.session_state["page"]),
    key="nav_radio",
)
st.session_state["page"] = page


# ── Page: About ─────────────────────────────────────────────────────────────

if page == "\U0001F30D About":

    # ── Hero section ──
    st.markdown(
        """
        <div style="text-align: center; padding: 0; margin-top: -1rem;">
        <span style="font-size: 4em;">\U0001F343</span>
        <h1 style="margin: 0; font-size: 2.5em;">OneLeafLeft</h1>
        <p style="font-size: 1.3em; color: #555; margin-top: 0.2em;">
        Predicting where deforestation will strike next in the Congo Basin
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

   

    # ── Quick navigation buttons ──
    nav_cols = st.columns(4)
    if nav_cols[0].button("\U0001F5FA\uFE0F  Risk Map", use_container_width=True):
        st.session_state["page"] = "\U0001F5FA\uFE0F Risk Map"
        st.rerun()
    if nav_cols[1].button("\U0001F4CA  Performance", use_container_width=True):
        st.session_state["page"] = "\U0001F4CA Performance"
        st.rerun()
    if nav_cols[2].button("\U0001F50D  Key Drivers", use_container_width=True):
        st.session_state["page"] = "\U0001F50D Key Drivers"
        st.rerun()
    if nav_cols[3].button("\U0001F9EA  Experiments", use_container_width=True):
        st.session_state["page"] = "\U0001F9EA Experiments"
        st.rerun()

    st.markdown("")

    # ── Mini preview map ──
    pred_hero = load_predictions()
    hero_sample = pred_hero.nlargest(2000, "proba").copy()
    hero_sample["risk_pct"] = (hero_sample["proba"] * 100).round(1)
    fig_hero = px.scatter_mapbox(
        hero_sample,
        lat="lat", lon="lon",
        color="risk_pct",
        color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c", "#8b0000"],
        range_color=[hero_sample["risk_pct"].quantile(0.05), hero_sample["risk_pct"].max()],
        hover_data={"risk_pct": ":.1f", "lat": ":.3f", "lon": ":.3f"},
        labels={"risk_pct": "Risk (%)"},
        zoom=5.5, height=400,
        mapbox_style="carto-positron",
        center={"lat": 2.1, "lon": 25.0},
    )
    fig_hero.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title="Risk %", len=0.5),
    )
    st.plotly_chart(fig_hero, use_container_width=True)
    st.caption(
        "Preview: 2,000 highest-risk locations. "
        "Go to **Risk Map** for the full interactive map."
    )

     # ── Key numbers ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("\U0001F30D Locations", f"{info['val_n']:,}")
    col2.metric("\U0001F4CA Accuracy (AUC)", f"{info['val_auc_roc']:.1%}")
    col3.metric("\U0001F333 Features", f"{info['n_features']}")
    col4.metric("\U0001F4C5 Prediction year", "2024")

    st.markdown("")
    
    st.markdown("---")

    # ── Two-column layout: Why + What ──
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

            By screening just the **top 5%** highest-risk areas,
            we capture **~75% of actual deforestation** —
            **15\u00d7 more efficiently** than random monitoring.
            """
        )

    st.markdown("---")

    # ── Data sources ──
    st.markdown("### Data sources")
    st.markdown("The model combines **33 variables** from satellite imagery and public databases:")

    data_cols = st.columns(4)
    with data_cols[0]:
        st.markdown(
            """
            **\U0001F6F0\uFE0F Satellite**
            - Forest cover & loss history
            - Neighbourhood deforestation
            - Night light intensity

            *Hansen GFC, VIIRS*
            """
        )
    with data_cols[1]:
        st.markdown(
            """
            **\U0001F3D7\uFE0F Infrastructure**
            - Distance to roads
            - Distance to villages
            - Protected area status

            *OpenStreetMap, WDPA*
            """
        )
    with data_cols[2]:
        st.markdown(
            """
            **\U0001F30D Environment**
            - Elevation & slope
            - Temperature & rainfall
            - Dry season length

            *SRTM, ERA5, CHIRPS*
            """
        )
    with data_cols[3]:
        st.markdown(
            """
            **\U0001F465 Socio-economic**
            - Population density
            - Population growth
            - Governance indicators

            *WorldPop, World Bank*
            """
        )

    st.markdown("---")

    # ── How it works ──
    st.markdown("### How it works")
    steps = st.columns(5)
    step_icons = ["\U0001F4E1", "\U0001F504", "\U0001F6E1\uFE0F", "\U0001F916", "\U0001F50D"]
    step_titles = ["Extract", "Encode", "Split", "Train", "Explain"]
    step_descs = [
        "~140 features per location from Google Earth Engine",
        "Relative lags (*last year*, *2 years ago*) instead of absolute dates",
        "Train on 2016\u20132022, predict 2024 — no future leakage",
        "XGBoost gradient boosting — fast, accurate, interpretable",
        "SHAP values reveal *why* each location is at risk",
    ]
    for col, icon, title, desc in zip(steps, step_icons, step_titles, step_descs):
        col.markdown(
            f"""
            <div style="text-align: center;">
            <span style="font-size: 2em;">{icon}</span><br>
            <b>{title}</b><br>
            <small style="color: #666;">{desc}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Key findings ──
    st.markdown("### Key findings")
    f1, f2 = st.columns(2)
    with f1:
        st.success(
            "**Deforestation is contagious** \u2014 the strongest predictor is whether "
            "*nearby* forest was recently deforested. This contagion effect operates "
            "from 150m to 5km."
        )
        st.info(
            "**Roads and settlements matter** \u2014 proximity to human infrastructure "
            "is the second most important factor after spatial contagion."
        )
    with f2:
        st.warning(
            "**1 year of history is enough** \u2014 looking back further than 1\u20132 years "
            "barely improves predictions. Deforestation contagion is nearly instantaneous."
        )
        st.error(
            "**Climate and economics add little** \u2014 while relevant for understanding "
            "deforestation, they don't improve *prediction* beyond spatial contagion."
        )

    st.markdown("---")

    # ── Reading guide ──
    with st.expander("How to read the results (important for non-specialists)"):
        st.markdown(
            """
            Deforestation is a **very rare event** \u2014 only **0.36%** of locations are
            deforested in any given year. This has important consequences:

            - **The model ranks locations by risk**, it doesn't give a simple yes/no.
              A location with a 10% risk score is *not* certain to be deforested,
              but it is **28\u00d7 more likely** than an average location.

            - **Traditional metrics can be misleading.** Even a perfect model would
              produce many "false alarms" because there are so few actual deforestation events
              compared to the vast forest. This is normal for rare-event prediction.

            - **The right way to evaluate**: if we screen areas from highest to lowest risk,
              how quickly do we find the deforestation? Our model captures **75% of deforestation
              by screening just 5%** of the area. That's the real value.

            - **Think of it like a medical screening test**: you wouldn't expect a cancer
              screening to be perfect, but you'd want it to prioritise the right patients
              for follow-up. This model does the same for forests.
            """
        )

    # ── Footer ──
    st.markdown(
        """
        <div style="text-align: center; color: #888; font-size: 0.85em; margin-top: 2em;">
        <i>Guillaume Laumier & Marc Bouvier</i> \u2014 Universit\u00e9 Grenoble Alpes / NitiDae<br>
        Data: Google Earth Engine, OpenStreetMap, World Bank | Model: XGBoost + SHAP<br>
        <br>
        \u2190 <b>Use the sidebar</b> to explore the risk map, performance, key drivers, and experiments.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Page: Risk Map ──────────────────────────────────────────────────────────

elif page == "\U0001F5FA\uFE0F Risk Map":
    st.markdown(
        """
        # \U0001F5FA\uFE0F Deforestation Risk Map
        Each dot represents a forest location. **Red dots** have a high probability of
        being deforested. The model predicts whether each location will lose its forest
        cover in 2024, based on patterns observed in previous years.
        """
    )

    pred = load_predictions()
    threshold = info["threshold"]

    # Simple view selector
    view = st.radio(
        "What would you like to see?",
        [
            "Where is deforestation most likely?",
            "How well did the model predict?",
            "All locations coloured by risk",
        ],
        horizontal=True,
    )

    if view == "Where is deforestation most likely?":
        # Top risk locations
        n_show = st.slider("Number of high-risk locations", 500, 10000, 3000, step=500)
        df_plot = pred.nlargest(n_show, "proba").copy()
        df_plot["risk_pct"] = (df_plot["proba"] * 100).round(1)

        fig = px.scatter_mapbox(
            df_plot,
            lat="lat", lon="lon",
            color="risk_pct",
            color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c", "#8b0000"],
            range_color=[df_plot["risk_pct"].quantile(0.05), df_plot["risk_pct"].max()],
            hover_data={"risk_pct": ":.1f", "lat": ":.3f", "lon": ":.3f"},
            labels={"risk_pct": "Risk (%)"},
            zoom=6, height=650,
            mapbox_style="carto-positron",
            center={"lat": 5.1, "lon": 25.0},
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

        st.caption(
            f"Showing the **{n_show:,}** locations with highest predicted deforestation risk. "
            f"Risk ranges from {df_plot['risk_pct'].min():.1f}% to {df_plot['risk_pct'].max():.1f}%. "
            f"Use mouse wheel to zoom, click and drag to pan."
        )

    elif view == "How well did the model predict?":
        # Show actual deforestation vs predictions
        st.markdown(
            """
            The map below shows locations where deforestation **actually happened** in 2024.
            - \U0001F534 **Red**: the model correctly predicted deforestation
            - \U0001F535 **Blue**: the model missed the deforestation (false negative)
            """
        )

        defo = pred[pred["target"] == 1].copy()
        defo["status"] = np.where(
            defo["proba"] >= threshold,
            "Correctly predicted",
            "Missed by model",
        )

        fig = px.scatter_mapbox(
            defo,
            lat="lat", lon="lon",
            color="status",
            color_discrete_map={
                "Correctly predicted": "#e74c3c",
                "Missed by model": "#3498db",
            },
            hover_data={"proba": ":.3f"},
            labels={"proba": "Risk score"},
            zoom=6, height=650,
            mapbox_style="carto-positron",
            center={"lat": 5.1, "lon": 25.0},
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

        n_detected = (defo["status"] == "Correctly predicted").sum()
        n_missed = (defo["status"] == "Missed by model").sum()
        st.markdown(
            f"Out of **{len(defo):,}** deforested locations: "
            f"**{n_detected:,}** detected ({n_detected/len(defo)*100:.0f}%), "
            f"**{n_missed:,}** missed ({n_missed/len(defo)*100:.0f}%)."
        )

    else:
        # All points colored
        n_show = st.slider("Number of locations to display", 5000, 50000, 15000, step=5000)
        df_plot = pred.sample(min(n_show, len(pred)), random_state=42).copy()
        df_plot["risk_pct"] = (df_plot["proba"] * 100).round(1)

        fig = px.scatter_mapbox(
            df_plot,
            lat="lat", lon="lon",
            color="risk_pct",
            color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c"],
            range_color=[0, df_plot["risk_pct"].quantile(0.99)],
            hover_data={"risk_pct": ":.1f"},
            labels={"risk_pct": "Risk (%)"},
            zoom=6, height=650,
            mapbox_style="carto-positron",
            center={"lat": 5.1, "lon": 25.0},
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    # Point detail panel
    with st.expander("\U0001F50D Inspect a specific location"):
        st.markdown("Select a high-risk location to understand **why** the model thinks it's at risk.")

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


# ── Page: Performance ────────────────────────────────────────────────────────

elif page == "\U0001F4CA Performance":
    st.markdown(
        """
        # \U0001F4CA Model Performance

        How well does the model identify areas at risk of deforestation?
        Deforestation is a **rare event** (only 0.36% of locations), which makes
        prediction challenging.
        """
    )

    pred = load_predictions()
    y_true = pred["target"].values
    y_proba = pred["proba"].values
    threshold = info["threshold"]

    # Key metrics in plain language
    total_pos = y_true.sum()
    y_pred = (y_proba >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()

    st.info(
        f"**Context**: Only **{y_true.mean():.2%}** of locations are actually deforested "
        f"({int(y_true.sum()):,} out of {len(y_true):,}). "
        "With such a rare event, traditional metrics like recall and false alarm rate "
        "are misleading — what matters is **how efficiently the model ranks risk**. "
        "See the screening efficiency chart below."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Ranking quality (AUC-ROC)",
        f"{info['val_auc_roc']:.1%}",
        "High-risk areas are correctly ranked",
    )
    # Compute lift at top 5%
    top5_idx = int(len(y_proba) * 0.05)
    top5_order = np.argsort(-y_proba)[:top5_idx]
    top5_captured = y_true[top5_order].sum()
    col2.metric(
        "Top 5% screening",
        f"{top5_captured / total_pos:.0%} captured",
        f"{top5_captured:,}/{int(total_pos):,} deforestation events",
    )
    col3.metric(
        "Efficiency vs random",
        f"{(top5_captured / total_pos) / 0.05:.0f}x better",
        "Than screening areas randomly",
    )

    with st.expander("Threshold-based metrics (at model's optimal threshold)"):
        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("Detected", f"{tp}/{tp+fn}", f"{tp/(tp+fn)*100:.0f}% recall")
        tc2.metric("False alarms", f"{fp:,}",
                    f"{fp/(tp+fp)*100:.0f}% of alerts" if tp + fp > 0 else "N/A",
                    delta_color="inverse")
        tc3.metric("Threshold", f"{threshold:.3f}",
                    "Optimised on training data")
        st.caption(
            "These numbers look alarming because deforestation is extremely rare (0.36%). "
            "Even a good model produces many false alarms at any fixed threshold. "
            "The screening efficiency below is a more meaningful way to evaluate the model."
        )

    st.markdown("---")

    # Lift curve — the most intuitive metric
    st.subheader("Screening efficiency")
    st.markdown(
        """
        If we rank all locations by risk score and check them from highest to lowest,
        how much deforestation do we capture?
        """
    )

    order = np.argsort(-y_proba)
    y_sorted = y_true[order]
    cumsum = np.cumsum(y_sorted)
    total_pos = y_true.sum()
    n = len(y_true)
    pct_screened = np.arange(1, n + 1) / n * 100
    pct_captured = cumsum / total_pos * 100

    # Key milestones
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

    # Plot
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
        "The green curve shows that by screening just a small fraction of the area "
        "(starting from highest risk), we can capture most of the deforestation. "
        "The dashed line shows what random screening would achieve."
    )

    # More details in expander
    with st.expander("Technical details"):
        from sklearn.metrics import roc_curve, precision_recall_curve

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
                xaxis_title="Recall (% deforestation detected)",
                yaxis_title="Precision (% alerts correct)",
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

        st.markdown(
            f"**Confusion matrix** at threshold {threshold:.2f}:"
        )
        cm_df = pd.DataFrame(
            [[f"{tn:,}", f"{fp:,}"], [f"{fn:,}", f"{tp:,}"]],
            index=["Actually safe", "Actually deforested"],
            columns=["Predicted safe", "Predicted at risk"],
        )
        st.dataframe(cm_df, use_container_width=False)


# ── Page: Key Drivers ────────────────────────────────────────────────────────

elif page == "\U0001F50D Key Drivers":
    st.markdown(
        """
        # \U0001F50D What drives deforestation risk?

        The model identified the most important factors for predicting
        deforestation. Each bar shows how much influence a factor has
        on the predictions, on average.
        """
    )

    imp = load_feature_importance()

    # Add human labels and groups
    imp["label"] = imp["feature"].apply(humanize)
    imp["group"] = imp["feature"].apply(classify_feature)
    imp["group_label"] = imp["group"].map(GROUP_LABELS).fillna(imp["group"])

    # Group-level chart first (simpler)
    st.subheader("By category")
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
        xaxis_title="Total importance",
        height=350,
        margin=dict(l=0, r=0, t=5, b=0),
    )
    st.plotly_chart(fig_group, use_container_width=True)

    st.markdown(
        """
        > **Key insight**: Deforestation is **contagious** — the strongest predictor
        > of whether a forest will be cleared is whether **nearby forests** have
        > recently been cleared. This spatial contagion effect is far stronger than
        > any other factor (roads, population, climate, etc.).
        """
    )

    # Top individual features
    st.markdown("---")
    st.subheader("Top individual factors")

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

    # Explanations
    st.markdown(
        """
        #### What these factors mean

        - **Neighbourhood deforestation** (150m to 5km): if forests nearby have been
          cleared recently, this forest is at high risk. Deforestation spreads like
          a wave from cleared areas.

        - **Initial forest cover** (year 2000): dense primary forests are targeted
          differently than degraded forests.

        - **Distance to roads & villages**: forests close to roads and settlements
          are more accessible and therefore more vulnerable.

        - **Protected area status**: being inside a national park reduces risk,
          but doesn't eliminate it.
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


# ── Page: Experiments ────────────────────────────────────────────────────────

elif page == "\U0001F9EA Experiments":
    st.markdown(
        """
        # \U0001F9EA Scientific experiments

        We systematically tested which information the model actually needs.
        This "ablation study" removes groups of features to measure their
        contribution.
        """
    )

    # Feature groups
    st.subheader("Which data sources matter?")
    st.markdown(
        """
        We trained the model using only one type of information at a time.
        The chart shows how well the model performs with each group alone.
        """
    )

    fg_data = load_ablation("ablation_results_20260307.json")
    if fg_data:
        results = fg_data["results"]
        # Only show individual groups (not cumulative — too detailed)
        indiv = [r for r in results if r["name"].startswith("only_")]
        indiv_df = pd.DataFrame(indiv)

        # Human-readable names
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
            """
            > **Finding**: Neighbourhood deforestation alone (no other data needed)
            > achieves **94.7%** accuracy. Adding forest cover history and infrastructure
            > pushes it to **95.5%**. Adding more data beyond that doesn't help
            > — in fact, it slightly *hurts* performance.
            """
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

    # Temporal depth
    st.markdown("---")
    st.subheader("How much history does the model need?")
    st.markdown(
        "We tested whether the model benefits from looking further back in time."
    )

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
                yaxis_title="Prediction accuracy",
                yaxis_range=[0.9, 0.97],
                height=300,
                margin=dict(t=5),
            )
            st.plotly_chart(fig_tw, use_container_width=True)

            st.markdown(
                """
                > **Finding**: Just **1 year** of historical data is almost as good as
                > 4 years. Deforestation contagion is an *instantaneous* process —
                > what happened last year matters, but the year before barely adds
                > information. This contradicts the common assumption that
                > "more history = better predictions".
                """
            )

    # Spatial scale
    st.markdown("---")
    st.subheader("At what distance does deforestation spread?")
    st.markdown(
        "We tested different neighbourhood sizes to measure how far "
        "the 'contagion effect' of deforestation reaches."
    )

    spat_data = load_ablation("spatial_ablation_20260307.json")
    if spat_data:
        buf_results = [r for r in spat_data if r.get("experiment") == "buffer_radius"]
        indiv_buf = [r for r in buf_results if "+" not in r.get("label", "")]
        if indiv_buf:
            sb_df = pd.DataFrame(indiv_buf)
            radius_map = {
                "only_150m": "150m (nearby)",
                "only_500m": "500m",
                "only_1500m": "1.5km",
                "only_5000m": "5km (landscape)",
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
                """
                > **Finding**: The **150m radius** (immediate neighbourhood) gives the
                > best precision for detecting deforestation. Combining 150m with 1.5km
                > gives the overall best results — capturing both micro-local contagion
                > and landscape-level patterns.
                """
            )

        # Decay
        decay_results = [r for r in spat_data if r.get("experiment") == "spatial_decay"]
        if decay_results and decay_results[0].get("decay_data"):
            with st.expander("Spatial decay curve"):
                dd = decay_results[0]["decay_data"]
                radii = sorted(dd.keys(), key=lambda x: int(x))
                decay_df = pd.DataFrame([
                    {"Distance (m)": int(r),
                     "Mean deforestation rate": dd[r]["mean"],
                     "% locations affected": dd[r]["pct_nonzero"]}
                    for r in radii
                ])
                fig_decay = px.line(
                    decay_df, x="Distance (m)", y="Mean deforestation rate",
                    markers=True, log_x=True, height=300,
                )
                fig_decay.update_layout(margin=dict(t=5))
                st.plotly_chart(fig_decay, use_container_width=True)


# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888; font-size: 0.85em;">
    \U0001F343 <b>OneLeafLeft</b> — Deforestation prediction in the Congo Basin<br>
    XGBoost model trained on 730K observations | 250K locations | 2016-2024 data<br>
    <i>Guillaume Laumier & Marc Bouvier</i> | UGA / NitiDae
    </div>
    """,
    unsafe_allow_html=True,
)
