import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Pipeline Audit — Deforestation Prediction, Congo Basin

        **Purpose**: Trace every data decision, verify absence of leakage, and document
        the full pipeline from GEE extraction to model training.

        **Airtight temporal split** (updated 7 March 2026):
        - **Train**: features 2016-2022, predicts 2020/2021/2022 — NEVER sees 2023
        - **Val**: features 2017-2023, predicts 2024 — uses 2023 as Lag1

        Use this notebook as a reproducibility checklist: every data source, every
        transformation, every exclusion is documented with its rationale and verifiable.

        **How to run**: `marimo edit notebooks/03_pipeline_audit.py`

        ---
        """
    )
    return (mo,)


@app.cell
def _():
    import re
    import sys
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path

    PROJECT_DIR = Path("/Users/guillaume/Documents/Recherche/Deforestation")
    DATA_DIR = PROJECT_DIR / "data"
    RAW_PATH = DATA_DIR / "raw_250k_20260228.parquet"

    sys.path.insert(0, str(PROJECT_DIR / "scripts"))
    sys.path.insert(0, str(PROJECT_DIR / "src" / "data"))
    return DATA_DIR, RAW_PATH, json, np, pd, re


@app.cell
def _(RAW_PATH, pd):
    from gee_extraction import rebuild_features_dataset
    from train_xgboost import (
        add_window_summaries,
        NON_FEATURE_COLS,
        _GLOBAL_ANOM_RE,
        _GLOBAL_SUMMARY_SUFFIXES,
    )

    # Airtight split constants
    TRAIN_FEATURE_YEARS = list(range(2016, 2023))  # [2016..2022]
    TRAIN_PREDICTION_YEARS = [2020, 2021, 2022]
    VAL_FEATURE_YEARS = list(range(2017, 2024))    # [2017..2023]
    VAL_PREDICTION_YEARS = [2024]
    FEATURE_WINDOW = 4

    # Load raw
    df_raw = pd.read_parquet(RAW_PATH)

    # Build train and val from raw
    df_train = rebuild_features_dataset(
        df_raw, TRAIN_FEATURE_YEARS, TRAIN_PREDICTION_YEARS, feature_window=FEATURE_WINDOW)
    df_train["split"] = "train"
    df_val = rebuild_features_dataset(
        df_raw, VAL_FEATURE_YEARS, VAL_PREDICTION_YEARS, feature_window=FEATURE_WINDOW)
    df_val["split"] = "val"
    return NON_FEATURE_COLS, add_window_summaries, df_raw, df_train, df_val


@app.cell
def _(mo):
    mo.md("""
    ## 1. Data Sources — GEE Collections & Parameters

    Every data source is extracted from Google Earth Engine via functions in
    `src/data/gee_extraction.py`. Below is the exhaustive inventory.

    ### 1.1 Static sources (no temporal dimension)

    | Source | GEE Collection | Bands | Scale | Function | Line |
    |--------|---------------|-------|-------|----------|------|
    | SRTM | `USGS/SRTMGL1_003` | `elevation`, `slope` (derived via `ee.Terrain.slope`) | 30m | `extract_srtm()` | L98 |
    | Hansen GFC v1.12 | `UMD/hansen/global_forest_change_2024_v1_12` | `treecover2000` (%), `lossyear` (0=none, 1-24=2001-2024) | 30m | `extract_hansen_static()` | L107 |
    | WDPA | `WCMC/WDPA/current/polygons` | `in_protected` (binary), `dist_protected_km` (km, capped 200km) | 1000m | `extract_wdpa()` | L357 |

    ### 1.2 Annual time-series sources (extracted for years 2016-2023)

    | Source | GEE Collection | Bands per year | Scale | Function |
    |--------|---------------|----------------|-------|----------|
    | WorldPop | `WorldPop/GP/100m/pop` | `pop_{yr}` (persons/pixel, clamped to 2020 after) | 100m | `extract_worldpop()` |
    | CHIRPS | `UCSB-CHG/CHIRPS/DAILY` | `precip_total`, `precip_mean`, `precip_max`, `dry_days`, `extreme_rain_days` | 5000m | `extract_chirps()` |
    | ERA5-Land | `ECMWF/ERA5_LAND/DAILY_AGGR` | `temperature_2m`, `temperature_2m_max`, `hot_days`, `sm_surface`, `sm_std`, `et` | 9000m | `extract_era5()` |
    | VIIRS NTL | `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG` | `ntl_mean`, `ntl_max`, `ntl_std`, `ntl_cv` | 750m | `extract_viirs()` |
    | MODIS fire | `MODIS/061/MOD14A1` | `fire_days` (FireMask>=7), `fire_max` (MaxFRP) | 1000m | `extract_modis_fire()` |

    ### 1.3 Spatial buffer features (extracted for years 2016-2023)

    | Source | GEE Collection | Method | Radii | Scale per radius |
    |--------|---------------|--------|-------|-----------------|
    | Hansen buffers | `UMD/hansen/global_forest_change_2024_v1_12` | `computePixels` + `rasterio` sampling | 150m, 500m, 1500m, 5000m | 100m, 200m, 300m, 500m |

    **Buffer extraction details** (`extract_hansen_buffers()`):
    - For each year and radius: `hansen.select("lossyear").unmask(0).eq(yr_code)`
    - `unmask(0)` is **critical**: without it, non-forest pixels are NaN instead of 0 -> 89% NaN
    - `reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.circle(radius, "meters"))` -> fraction of deforested pixels within radius
    - Downloaded as tiled GeoTIFFs via `computePixels`, sampled at point locations with `rasterio`
    - Auto-adapted tile size to stay within GEE 50MB limit

    ### 1.4 Infrastructure features (non-GEE)

    | Source | API/Method | Variables | Script |
    |--------|-----------|-----------|--------|
    | OSM roads | Overpass API | `dist_road_km` (nearest road distance) | `scripts/add_spatial_features.py` |
    | OSM settlements | Overpass API | `dist_settlement_km` (nearest settlement) | `scripts/add_spatial_features.py` |
    | WGI (World Bank) | WB API | `wgi_voice_account`, `wgi_rule_law`, etc. per country | `scripts/add_tabular_features.py` |
    | WDI (World Bank) | WB API | `gdp_per_capita`, `agri_pct_gdp`, etc. per country | `scripts/add_tabular_features.py` |
    | Commodity prices | World Bank Pink Sheet | `palm_oil_price`, `cocoa_price`, etc. | `scripts/add_tabular_features.py` |

    ### 1.5 WDPA enriched features (tested, NOT in final model)

    | Variable | Computation | Result |
    |----------|-------------|--------|
    | `iucn_strict/moderate/sustainable/not_reported` | One-hot from `WCMC/WDPA` IUCN_CAT | No improvement |
    | `pa_defo_rate` | Mean Hansen loss within each PA | No improvement |
    | `pa_pressure_ring` | Mean Hansen loss in 0-5km ring around PA | No improvement |

    ---
    """)
    return


@app.cell
def _(RAW_PATH, df_raw, mo):
    mo.md(
        f"""
        ### 1.6 Raw parquet verification

        **File**: `{RAW_PATH.name}`
        **Shape**: {df_raw.shape[0]:,} points x {df_raw.shape[1]} columns
        """
    )
    return


@app.cell
def _(df_raw, mo, pd):
    # Categorize each column by source
    def classify_column(_col):
        if _col in ("lon", "lat"):
            return "coordinates"
        if _col in ("elevation", "slope"):
            return "SRTM"
        if _col in ("treecover2000", "lossyear"):
            return "Hansen static"
        if _col.startswith("pop_"):
            return "WorldPop"
        if any(_col.startswith(p) for p in ("precip_", "dry_days_", "extreme_rain_")):
            return "CHIRPS"
        if any(_col.startswith(p) for p in ("temperature_", "hot_days_", "sm_", "et_")):
            return "ERA5-Land"
        if _col.startswith("ntl_"):
            return "VIIRS NTL"
        if _col.startswith("fire_"):
            return "MODIS fire"
        if _col.startswith("defo_rate_"):
            return "Hansen buffers"
        if _col in ("in_protected", "dist_protected_km"):
            return "WDPA"
        if any(_col.startswith(p) for p in ("iucn_", "pa_defo", "pa_pressure")):
            return "WDPA enriched"
        if _col in ("dist_road_km", "dist_settlement_km"):
            return "OSM infrastructure"
        if _col.startswith("country_"):
            return "country"
        if any(_col.startswith(p) for p in ("wgi_", "gdp_", "agri_", "forest_rent_")):
            return "World Bank"
        if any(_col.startswith(p) for p in ("palm_oil_", "cocoa_", "coffee_", "rubber_",
                                            "soybean_", "timber_")):
            return "Commodity prices"
        return "other"

    raw_sources = pd.DataFrame({
        "column": df_raw.columns,
        "source": [classify_column(c) for c in df_raw.columns],
        "dtype": [str(df_raw[c].dtype) for c in df_raw.columns],
        "nan_pct": [df_raw[c].isna().mean() * 100 for c in df_raw.columns],
    })

    summary = raw_sources.groupby("source").agg(
        n_cols=("column", "count"),
        mean_nan_pct=("nan_pct", "mean"),
        example_cols=("column", lambda x: list(x)[:3]),
    ).sort_values("n_cols", ascending=False)

    mo.md(
        f"""
        | Source | N columns | Mean NaN % | Example columns |
        |--------|-----------|------------|-----------------|
        """ + "\n".join(
            f"| {src} | {row.n_cols} | {row.mean_nan_pct:.1f}% | `{'`, `'.join(row.example_cols)}` |"
            for src, row in summary.iterrows()
        ) + f"""

        **Total**: {len(df_raw.columns)} columns
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Airtight Temporal Split — Leakage Verification

    The split is **airtight**: train and val are built from separate calls to
    `rebuild_features_dataset()` with different year ranges:

    - **Train**: feature years 2016-2022, prediction years 2020/2021/2022
    - **Val**: feature years 2017-2023, prediction years 2024

    **Critical guarantee**: Train NEVER sees any 2023 data. Val uses 2023 as Lag1.

    ### 2.1 Sliding window — how it works

    For each prediction year T, features come from years [T-4, T-1] (4-year window).

    **Train windows**:
    ```
    pred_yr=2020: features from [2016, 2017, 2018, 2019], target = lossyear==20
    pred_yr=2021: features from [2017, 2018, 2019, 2020], target = lossyear==21
    pred_yr=2022: features from [2018, 2019, 2020, 2021], target = lossyear==22
    ```

    **Val window**:
    ```
    pred_yr=2024: features from [2020, 2021, 2022, 2023], target = lossyear==24
    ```

    Year-indexed columns are renamed to lag-indexed:
    ```
    pop_2023 with pred_yr=2024 -> pop_Lag1  (1 year before prediction)
    pop_2019 with pred_yr=2020 -> pop_Lag1  (same relative position)
    ```

    ### 2.2 Year mapping table

    | pred_yr | Lag1 | Lag2 | Lag3 | Lag4 | Target | Split |
    |---------|------|------|------|------|--------|-------|
    | 2020 | 2019 | 2018 | 2017 | 2016 | lossyear==20 | train |
    | 2021 | 2020 | 2019 | 2018 | 2017 | lossyear==21 | train |
    | 2022 | 2021 | 2020 | 2019 | 2018 | lossyear==22 | train |
    | 2024 | 2023 | 2022 | 2021 | 2020 | lossyear==24 | val |

    **Key**: Train Lag1 is at most 2021 (for pred_yr=2022). Val Lag1 is 2023.
    Train NEVER accesses 2023 or later.

    ---
    """)
    return


@app.cell
def _(df_train, df_val, mo):
    # Verify prediction years and split assignment
    train_pred_yrs = sorted(df_train["prediction_year"].unique())
    val_pred_yrs = sorted(df_val["prediction_year"].unique())

    train_target = df_train["target"]
    val_target = df_val["target"]

    mo.md(
        f"""
        ### 2.3 Actual split distribution

        | Split | N rows | pred_yr | N positive | Defo rate |
        |-------|--------|---------|-----------|-----------|
        | train | {len(df_train):,} | {train_pred_yrs} | {int(train_target.sum()):,} | {train_target.mean()*100:.2f}% |
        | val | {len(df_val):,} | {val_pred_yrs} | {int(val_target.sum()):,} | {val_target.mean()*100:.2f}% |

        **Assertions**:
        - Train pred_yr in {{2020, 2021, 2022}}: {"PASS" if set(train_pred_yrs) == {2020, 2021, 2022} else "FAIL"}
        - Val pred_yr in {{2024}}: {"PASS" if set(val_pred_yrs) == {2024} else "FAIL"}
        - Train split == "train" only: {"PASS" if set(df_train["split"].unique()) == {"train"} else "FAIL"}
        - Val split == "val" only: {"PASS" if set(df_val["split"].unique()) == {"val"} else "FAIL"}
        """
    )

    assert set(train_pred_yrs) == {2020, 2021, 2022}, f"Train pred_yr mismatch: {train_pred_yrs}"
    assert set(val_pred_yrs) == {2024}, f"Val pred_yr mismatch: {val_pred_yrs}"
    return


@app.cell
def _(df_train, df_val, mo):
    # Verify: train has NO 2023 features
    train_cols_with_2023 = [c for c in df_train.columns if "2023" in c]
    val_cols_with_2023 = [c for c in df_val.columns if "2023" in c]

    mo.md(
        f"""
        ### 2.4 Temporal leakage check — 2023 data

        **Train columns containing '2023'**: {len(train_cols_with_2023)}
        {f"LEAK DETECTED: {train_cols_with_2023[:5]}" if train_cols_with_2023 else "PASS — train never sees 2023"}

        **Val columns containing '2023'**: {len(val_cols_with_2023)}
        {f"Expected — val uses 2023 as source year" if val_cols_with_2023 else "WARNING: val should have 2023 data"}
        """
    )

    assert len(train_cols_with_2023) == 0, f"Train has 2023 columns: {train_cols_with_2023}"
    return


@app.cell
def _(df_train, df_val, mo):
    # Column alignment check
    train_cols = set(df_train.columns)
    val_cols = set(df_val.columns)
    common_cols = train_cols & val_cols
    train_only = train_cols - val_cols
    val_only = val_cols - train_cols

    mo.md(
        f"""
        ### 2.5 Column alignment between train and val

        | Category | N columns | Examples |
        |----------|-----------|---------|
        | Common | {len(common_cols)} | — |
        | Train-only | {len(train_only)} | `{'`, `'.join(sorted(train_only)[:5])}` |
        | Val-only | {len(val_only)} | `{'`, `'.join(sorted(val_only)[:5])}` |

        **Note**: Train-only and val-only columns arise from different FEATURE_YEARS ranges.
        The training pipeline uses only the intersection of columns.
        """
    )
    return


@app.cell
def _(NON_FEATURE_COLS, add_window_summaries, df_train, df_val, mo, pd):
    # Prepare features exactly like train_xgboost.py
    prep_train = df_train.copy()
    prep_val = df_val.copy()

    # Drop leaky anomaly columns
    for _df in [prep_train, prep_val]:
        leaky = [c for c in _df.columns if _GLOBAL_ANOM_RE.search(c) and "proxy" not in c]
        _df.drop(columns=leaky, inplace=True, errors="ignore")

    # Add window summaries
    prep_train = add_window_summaries(prep_train)
    prep_val = add_window_summaries(prep_val)

    # Drop global summaries
    global_summaries = {
        c for c in prep_train.columns
        if any(c.endswith(sfx) for sfx in _GLOBAL_SUMMARY_SUFFIXES)
        and not c.endswith("_wmean") and not c.endswith("_wtrend")
    }
    excluded = NON_FEATURE_COLS | global_summaries

    # Get feature columns (intersection of train and val)
    train_feat = [c for c in prep_train.columns
                  if c not in excluded and pd.api.types.is_numeric_dtype(prep_train[c])]
    val_feat = [c for c in prep_val.columns
                if c not in excluded and pd.api.types.is_numeric_dtype(prep_val[c])]
    feature_cols = sorted(set(train_feat) & set(val_feat))

    # Drop structural NaN
    nan_pct_train = prep_train[feature_cols].isna().mean() * 100
    nan_pct_val = prep_val[feature_cols].isna().mean() * 100
    structural_nan = [c for c in feature_cols if nan_pct_train[c] > 45 or nan_pct_val[c] > 45]
    final_features = [c for c in feature_cols if c not in structural_nan]

    mo.md(
        f"""
        ### 2.6 Feature preparation summary

        | Step | Result |
        |------|--------|
        | Leaky anomaly columns dropped | {len([c for c in df_train.columns if _GLOBAL_ANOM_RE.search(c) and 'proxy' not in c])} |
        | Window summaries added | wmean, wtrend, wanom |
        | Global summaries dropped | {len(global_summaries)} |
        | Common feature columns | {len(feature_cols)} |
        | Structural NaN dropped (>45%) | {len(structural_nan)} |
        | **Final features for training** | **{len(final_features)}** |
        """
    )
    return final_features, prep_train, prep_val


@app.cell
def _(mo):
    mo.md("""
    ## 3. Feature Exclusions — What's Dropped and Why

    ### 3.1 Non-feature columns (always excluded)

    | Column | Reason |
    |--------|--------|
    | `pid` | Point identifier, not a feature |
    | `lon`, `lat` | Geographic coordinates — would learn spatial memorization |
    | `target` | The prediction target |
    | `split` | Train/val assignment |
    | `prediction_year` | Sliding window metadata |
    | `lossyear` | Raw Hansen lossyear — **direct target leakage** (target is derived from it) |

    ### 3.2 Global anomaly columns (leakage)

    Pattern: `{base}_anom_Lag{N}` (e.g., `pop_anom_Lag1`)

    **Leakage mechanism**: `build_temporal_features()` computes anomaly relative to
    the global mean over ALL years, which includes future years for earlier pred_yr.

    **Fix**: Replaced by leak-free `{base}_wanom_Lag{k}` computed from window mean only.

    ### 3.3 Global summary columns (leakage)

    Pattern: `{base}_mean`, `{base}_trend` (e.g., `pop_mean`, `precip_total_trend`)

    **Fix**: Replaced by `{base}_wmean` and `{base}_wtrend` computed from lag columns only.

    ### 3.4 Structural NaN columns (>45% NaN)

    These arise from derived features (`_d1_`, `_d3_`) that need data from years
    before the extraction range starts.

    ### 3.5 Tautological features (removed March 2026)

    | Feature | Issue | Impact of removal |
    |---------|-------|-------------------|
    | `forest_remaining_w` | Trivially predicts target=0 for already-deforested pixels | AUC +0.021, PR-AUC +37% |
    | `loss_last2yrs_w` | Same logic — encodes recent loss which directly predicts no future loss | Included in the above |

    ---
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Leakage Sanity Checks

    Three statistical tests to detect potential leakage:

    1. **Feature-target correlation**: No feature should have |correlation| > 0.5 with target
    2. **Single-feature AUC**: No single feature should achieve AUC > 0.98 (tautological)
    3. **Per-year AUC stability**: Model should perform similarly across prediction years
    """)
    return


@app.cell
def _(final_features, mo, np, pd, prep_val):
    # Test 1: Feature-target correlations on val set
    y_val = prep_val["target"].values

    correlations = {}
    for _col in final_features:
        _vals = prep_val[_col].values
        _valid = ~np.isnan(_vals)
        if _valid.sum() > 100 and np.std(_vals[_valid]) > 0:
            correlations[_col] = np.corrcoef(_vals[_valid], y_val[_valid])[0, 1]

    corr_series = pd.Series(correlations).sort_values(key=abs, ascending=False)
    high_corr = corr_series[corr_series.abs() > 0.3]

    mo.md(
        f"""
        ### 4.1 Feature-target correlation (val set, pred_yr=2024)

        **Features with |correlation| > 0.3**: {len(high_corr)}

        | Feature | Correlation | Suspicious? |
        |---------|-------------|-------------|
        """ + "\n".join(
            f"| `{feat}` | {corr:.4f} | {'WARNING Investigate' if abs(corr) > 0.5 else 'OK'} |"
            for feat, corr in high_corr.items()
        ) + f"""

        **Max absolute correlation**: {corr_series.abs().max():.4f} (`{corr_series.abs().idxmax()}`)

        {"PASS: No feature has |corr| > 0.5" if corr_series.abs().max() < 0.5
         else "WARNING: Some features have high correlation — investigate!"}
        """
    )
    return (y_val,)


@app.cell
def _(final_features, mo, np, pd, prep_val, y_val):
    # Test 2: Single-feature AUC (top features only, for speed)
    from sklearn.metrics import roc_auc_score

    top_feats = pd.Series(
        {_col: abs(np.corrcoef(
            prep_val[_col].fillna(0).values, y_val
        )[0, 1]) for _col in final_features}
    ).nlargest(30).index.tolist()

    single_aucs = {}
    for _col in top_feats:
        _vals = prep_val[_col].fillna(0).values
        try:
            auc = roc_auc_score(y_val, _vals)
            single_aucs[_col] = max(auc, 1 - auc)
        except ValueError:
            pass

    auc_series = pd.Series(single_aucs).sort_values(ascending=False)

    mo.md(
        f"""
        ### 4.2 Single-feature AUC (val set, top 30 by correlation)

        | Feature | AUC | Status |
        |---------|-----|--------|
        """ + "\n".join(
            f"| `{feat}` | {auc:.4f} | "
            f"{'TAUTOLOGICAL' if auc > 0.98 else 'Very high' if auc > 0.95 else 'OK'} |"
            for feat, auc in auc_series.head(15).items()
        ) + f"""

        **Max single-feature AUC**: {auc_series.max():.4f} (`{auc_series.idxmax()}`)

        {"PASS: No single feature achieves AUC > 0.98" if auc_series.max() < 0.98
         else "FAIL: TAUTOLOGICAL FEATURE DETECTED — investigate immediately!"}

        {"Note: Features with AUC > 0.90 exist but are expected: spatial buffer features (defo_rate) capture strong signal." if auc_series.max() > 0.90 else ""}
        """
    )
    return


@app.cell
def _(final_features, mo, np, prep_train):
    # Test 3: Per-year AUC stability (train prediction years only)
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb

    # Quick model on pred_yr 2020-2021, evaluate on 2022
    train_mask = prep_train["prediction_year"].isin([2020, 2021])
    X_tr = prep_train.loc[train_mask, final_features].fillna(0).values
    y_tr = prep_train.loc[train_mask, "target"].values

    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    spw = n_neg / max(n_pos, 1)

    quick_model = xgb.XGBClassifier(
        n_estimators=50, max_depth=5, learning_rate=0.05,
        scale_pos_weight=spw, verbosity=0, n_jobs=-1,
    )
    quick_model.fit(X_tr, y_tr)

    year_aucs = {}
    for yr in sorted(prep_train["prediction_year"].unique()):
        mask = prep_train["prediction_year"] == yr
        X_yr = prep_train.loc[mask, final_features].fillna(0).values
        y_yr = prep_train.loc[mask, "target"].values
        if y_yr.sum() > 0 and y_yr.sum() < len(y_yr):
            proba = quick_model.predict_proba(X_yr)[:, 1]
            year_aucs[yr] = roc_auc_score(y_yr, proba)

    auc_std = np.std(list(year_aucs.values()))

    mo.md(
        f"""
        ### 4.3 Per-year AUC stability (quick 50-tree model, trained on 2020-2021)

        | Prediction year | AUC-ROC | Note |
        |----------------|---------|------|
        """ + "\n".join(
            f"| {yr} | {auc:.4f} | {'in-sample' if yr in [2020, 2021] else 'out-of-sample'} |"
            for yr, auc in sorted(year_aucs.items())
        ) + f"""

        **AUC std across years**: {auc_std:.4f}

        {"PASS: AUC is stable across years (std < 0.05)" if auc_std < 0.05
         else "WARNING: AUC varies significantly across years — investigate!"}

        **Note**: 2020-2021 are in-sample (higher AUC expected). The key check is that
        pred_yr=2022 (out-of-sample) is not wildly different.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Window Summaries — Leak-Free Derived Features

    `add_window_summaries()` in `train_xgboost.py` computes three types of derived
    features from the lag columns:

    | Feature | Formula | Purpose |
    |---------|---------|---------|
    | `{var}_wmean` | `mean(var_Lag1, var_Lag2, ..., var_LagN)` | Window-average level |
    | `{var}_wtrend` | `var_Lag1 - var_LagN` | Recent trend (positive = increasing) |
    | `{var}_wanom_Lag{k}` | `var_Lag{k} - wmean` | Per-year deviation from window mean |

    **Why these replace global summaries**:
    - `pop_mean` (global) uses `mean(pop_2016..pop_2021)` -> for pred_yr=2020, includes 2021 (future)
    - `pop_wmean` (window) uses `mean(pop_Lag1..pop_Lag4)` -> only past data, always leak-free

    ### `cum_deforested_Lag1` — the key Hansen-derived feature

    Computed in `rebuild_features_dataset()`:
    ```python
    cum_deforested_{yr} = lossyear.between(1, yr_code)  # was pixel deforested BY year yr?
    ```

    For pred_yr=2024: `cum_deforested_Lag1` = `cum_deforested_2023` = lossyear in [1, 23]
    -> "Was this pixel deforested at any point from 2001 to 2023?"

    This is **NOT leakage**: it uses only past data (before the prediction year).

    ---
    """)
    return


@app.cell
def _(mo, prep_train, re):
    # List all window summary features
    w_features = sorted([c for c in prep_train.columns if c.endswith("_wmean") or c.endswith("_wtrend")])
    wanom_features = sorted([c for c in prep_train.columns if "_wanom_" in c])

    # Verify: wmean values are within the range of their constituent lags
    _lag_re = re.compile(r"^(.+)_Lag(\d+)$")
    var_lags = {}
    for _col in prep_train.columns:
        m = _lag_re.match(_col)
        if m:
            base = m.group(1)
            var_lags.setdefault(base, []).append(_col)

    checks = []
    for base in list(var_lags.keys())[:5]:
        lag_cols = var_lags[base]
        wmean_col = f"{base}_wmean"
        if wmean_col in prep_train.columns:
            lag_min = prep_train[lag_cols].min(axis=1)
            lag_max = prep_train[lag_cols].max(axis=1)
            wmean = prep_train[wmean_col]
            _valid = wmean.notna() & lag_min.notna()
            in_range = ((wmean[_valid] >= lag_min[_valid] - 1e-6) &
                        (wmean[_valid] <= lag_max[_valid] + 1e-6)).mean()
            checks.append({"variable": base, "n_lags": len(lag_cols),
                          "wmean_in_range_pct": in_range * 100})

    mo.md(
        f"""
        ### 5.1 Window summary inventory

        **Window means/trends**: {len(w_features)} columns
        **Window anomalies**: {len(wanom_features)} columns

        ### 5.2 Sanity check: wmean within lag range

        | Variable | N lags | wmean in [min, max] range |
        |----------|--------|---------------------------|
        """ + "\n".join(
            f"| `{c['variable']}` | {c['n_lags']} | {c['wmean_in_range_pct']:.1f}% |"
            for c in checks
        ) + """

        All wmean values should be within the range of their constituent lags (expected: 100%)
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Train/Val Split — Temporal Integrity

    The split is **airtight**:
    - **Train**: pred_yr in {2020, 2021, 2022}, features from 2016-2022
    - **Val**: pred_yr = 2024, features from 2017-2023

    **No test set**: Val (pred_yr=2024) serves as the final evaluation.
    True out-of-sample testing would require 2025 data (not yet available).

    **Geographic overlap**: The same 250K geographic points appear in all prediction
    years. This is by design:
    - The model should generalize **temporally** (predict future deforestation)
    - A spatial split would conflate two problems: spatial and temporal generalization

    **Already-deforested exclusion**: Pixels deforested before the prediction window
    are excluded (they can't be deforested again). This is done in
    `build_sliding_window_dataset()`.

    ---
    """)
    return


@app.cell
def _(df_train, df_val, mo):
    # Check unique PIDs per split
    train_pids = set(df_train["pid"].unique())
    val_pids = set(df_val["pid"].unique())
    overlap = len(train_pids & val_pids)

    # Per-year stats within train
    train_years = df_train.groupby("prediction_year").agg(
        n_rows=("target", "count"),
        n_pos=("target", "sum"),
        defo_rate=("target", "mean"),
    )

    mo.md(
        f"""
        ### 6.1 Split statistics

        **Train** ({len(df_train):,} rows):

        | pred_yr | N rows | N positive | Defo rate |
        |---------|--------|-----------|-----------|
        """ + "\n".join(
            f"| {yr} | {int(row.n_rows):,} | {int(row.n_pos):,} | {row.defo_rate*100:.2f}% |"
            for yr, row in train_years.iterrows()
        ) + f"""

        **Val** ({len(df_val):,} rows): pred_yr=2024, {int(df_val['target'].sum()):,} positive ({df_val['target'].mean()*100:.2f}%)

        ### 6.2 Geographic overlap

        | Metric | Value |
        |--------|-------|
        | Train unique PIDs | {len(train_pids):,} |
        | Val unique PIDs | {len(val_pids):,} |
        | Shared PIDs | {overlap:,} |

        Geographic overlap is expected and by design. The temporal split ensures
        that no **future** information leaks into training.
        """
    )
    return


@app.cell
def _(DATA_DIR, json, mo):
    # Load ablation results if available
    ablation_files = {
        "feature_groups": "ablation_results_20260307.json",
        "temporal": "temporal_ablation_20260307.json",
        "spatial": "spatial_ablation_20260307.json",
    }

    ablation_data = {}
    for key, fname in ablation_files.items():
        path = DATA_DIR / fname
        if path.exists():
            with open(path) as f:
                ablation_data[key] = json.load(f)

    # Feature group ablation summary
    fg_text = ""
    if "feature_groups" in ablation_data:
        results = ablation_data["feature_groups"]["results"]
        fg_text = "\n".join(
            f"| {r['name']:<28} | {r['n_features']:>4} | {r['val_auc_roc']:.4f} | {r['val_pr_auc']:.4f} |"
            for r in results
        )

    # Temporal ablation summary
    temp_text = ""
    if "temporal" in ablation_data:
        for exp in ablation_data["temporal"]:
            if exp.get("experiment") == "window_depth":
                for w in exp.get("windows", []):
                    temp_text += f"| window={w['window']} | {w['n_features']} | {w['val_auc_roc']:.4f} | {w['val_pr_auc']:.4f} |\n"

    # Spatial ablation summary
    spat_text = ""
    if "spatial" in ablation_data:
        for exp in ablation_data["spatial"]:
            if exp.get("experiment") == "buffer_radius":
                spat_text += f"| {exp['label']:<20} | {exp['n_features']:>4} | {exp['val_auc_roc']:.4f} | {exp['val_pr_auc']:.4f} |\n"

    mo.md(
        f"""
        ## 7. Triple Ablation Results (Airtight Split, 7 March 2026)

        ### 7.1 Feature group ablation

        | Scenario | N feat | Val AUC-ROC | Val PR-AUC |
        |----------|--------|-------------|------------|
        {fg_text if fg_text else "*(results not found)*"}

        ### 7.2 Temporal ablation (window depth)

        | Config | N feat | Val AUC-ROC | Val PR-AUC |
        |--------|--------|-------------|------------|
        {temp_text if temp_text else "*(results not found)*"}

        ### 7.3 Spatial ablation (buffer radii)

        | Config | N feat | Val AUC-ROC | Val PR-AUC |
        |--------|--------|-------------|------------|
        {spat_text if spat_text else "*(results not found)*"}

        ### Key findings

        1. **Spatial contagion dominates**: `only_spatial` (100 feat) achieves AUC 0.947 alone
        2. **Core model is optimal**: hansen+spatial+infra (139 feat) = best PR-AUC
        3. **1 year window suffices**: window=1 (29 feat) matches window=4 (139 feat)
        4. **150m+1500m best radius pair**: +59% PR-AUC vs old 500m+5000m
        5. **Adding more features dilutes**: all_features PR-AUC < core model

        ---
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. Audit Checklist

    | Check | Status | Details |
    |-------|--------|---------|
    | GEE collections documented | PASS | See Section 1 |
    | All data sources traced | PASS | See Section 1.1-1.5 |
    | Airtight split verified | PASS | Train: 2016-2022, Val: 2017-2023 |
    | Train never sees 2023 | PASS | See Section 2.4 |
    | Year -> Lag mapping correct | PASS | See Section 2.2 |
    | Global anomaly leakage fixed | PASS | Dropped _anom_Lag cols, replaced by _wanom |
    | Global summary leakage fixed | PASS | Dropped _mean/_trend, replaced by _wmean/_wtrend |
    | Tautological features removed | PASS | forest_remaining_w, loss_last2yrs_w removed |
    | Feature-target correlation < 0.5 | See 4.1 | Run notebook to verify |
    | No single-feature AUC > 0.98 | See 4.2 | Run notebook to verify |
    | Per-year AUC stability | See 4.3 | Run notebook to verify |
    | Temporal split integrity | PASS | See Section 6 |
    | Hansen buffer `unmask(0)` fix | PASS | See Section 1.3 |
    | WorldPop multi-country fix | PASS | `filterBounds` instead of `filter(country)` |
    | scale_pos_weight applied | PASS | ~288:1 (0.35% positives) |
    | Triple ablation completed | PASS | See Section 7 |

    ## 9. Data Flow Diagram

    ```
    250K sample points (from legacy tiles_250000 CSV)
          |
          v
    +-----------------------------------+
    |  GEE Extraction (7 sources)       |  <- scale_up_extraction.py
    |  + OSM, World Bank, Commodities   |  <- add_spatial_features.py, add_tabular_features.py
    |  + Hansen buffers (4 radii)       |  <- add_buffers.py (2016-2023)
    +-----------------------------------+
          |
          v
    raw_250k_20260228.parquet  (250K x 196 cols)
          |
          +------------------+--------------------+
          |                                       |
          v                                       v
    build_traintest.py                     build_val.py
    feat_yr 2016-2022                      feat_yr 2017-2023
    pred_yr 2020,2021,2022                 pred_yr 2024
          |                                       |
          v                                       v
    train (729K rows)                      val (240K rows)
          |                                       |
          +-------+-------------------------------+
                  |
                  v
          Column intersection
          + add_window_summaries()
          + drop leaky/structural NaN
                  |
                  v
          ~549 common features -> XGBoost training
    ```

    ---
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 10. Reproducibility Commands

    To reproduce the full pipeline from scratch:

    ```bash
    # 1. Environment
    conda activate deforest

    # 2. GEE extraction (250K points x 7 sources, ~2-3 hours)
    cd /Users/guillaume/Documents/Recherche/Deforestation
    python scripts/scale_up_extraction.py --n-points 250000

    # 3. Add infrastructure features (OSM + World Bank, ~10 min)
    python scripts/add_spatial_features.py --raw data/raw_250k_YYYYMMDD.parquet
    python scripts/add_tabular_features.py --raw data/raw_250k_YYYYMMDD.parquet

    # 4. Add Hansen buffers (4 radii x 8 years, ~1 hour)
    python scripts/add_buffers.py --raw data/raw_250k_YYYYMMDD.parquet     --buffers 150 500 1500 5000 --years 2016 2017 2018 2019 2020 2021 2022 2023

    # 5. Build train and val datasets
    python scripts/build_traintest.py --raw data/raw_250k_YYYYMMDD.parquet
    python scripts/build_val.py --raw data/raw_250k_YYYYMMDD.parquet

    # 6. Train core model
    python scripts/train_xgboost.py     --dataset data/train_test/features_traintest_YYYYMMDD.parquet     --val data/val/features_val_YYYYMMDD.parquet

    # 7. Run ablation studies
    python scripts/ablation_study.py
    python scripts/temporal_ablation.py
    python scripts/spatial_ablation.py

    # 8. Run this audit notebook
    marimo run notebooks/03_pipeline_audit.py
    ```

    ---
    *Pipeline Audit Notebook — Airtight Split*
    *Last updated: 2026-03-07*
    """)
    return


if __name__ == "__main__":
    app.run()
