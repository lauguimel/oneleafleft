"""Mini pipeline: 1K points × 4 GEE sources → XGBoost → SHAP.

Step 1 validation: end-to-end chain before scaling to 250K.

Sources:
    - Hansen GFC v1.12 (treecover2000, lossyear → per-year binary)
    - SRTM (elevation, slope)
    - WorldPop (population density, annual 2017-2021)
    - CHIRPS daily → annual profiles (precipitation stats, 2017-2021)

Target: P(deforestation in 2022)

Usage:
    conda activate deforest
    python scripts/mini_pipeline.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

import ee
from data.gee_utils import init_gee

LEGACY_CSV = Path(
    "/Users/guillaume/Documents/Clouds/UGA/Recherche/deforestation"
    "/src/data/tiles_250000_10N_020E_20231023.csv"
)
OUTPUT_DIR = PROJECT_DIR / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

YEARS = [2017, 2018, 2019, 2020, 2021]  # features
TARGET_YEAR = 2022                        # prediction target
N_POINTS = 1000
SEED = 42

# ─── 1. Initialize GEE ───────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1 — Initialize GEE")
print("=" * 60)
init_gee()
print("  GEE OK")

# ─── 2. Stratified sample of 1K points ───────────────────────────────────────

print("\nSTEP 2 — Load and sample points")
tiles = pd.read_csv(
    LEGACY_CSV,
    usecols=["longitude", "latitude", "lossyear_22_mean",
             "treecover2000_mean", "SRTM_mean"],
)
print(f"  {len(tiles):,} total points")

# Binary label: any deforestation in 2022?
tiles["deforested_2022"] = (tiles["lossyear_22_mean"] > 0).astype(int)
print(f"  Deforested 2022: {tiles['deforested_2022'].sum():,} ({tiles['deforested_2022'].mean()*100:.1f}%)")

# Stratified sample: 50% each class
pos = tiles[tiles["deforested_2022"] == 1].sample(N_POINTS // 2, random_state=SEED)
neg = tiles[tiles["deforested_2022"] == 0].sample(N_POINTS // 2, random_state=SEED)
sample = pd.concat([pos, neg]).sample(frac=1, random_state=SEED).reset_index(drop=True)
print(f"  Sample: {len(sample)} points ({sample['deforested_2022'].sum()} positive, {(~sample['deforested_2022'].astype(bool)).sum()} negative)")

# ─── 3. Build GEE FeatureCollection ──────────────────────────────────────────

print("\nSTEP 3 — Upload points to GEE")
t0 = time.time()

features = [
    ee.Feature(
        ee.Geometry.Point([float(row.longitude), float(row.latitude)]),
        {"pid": int(i), "lon": float(row.longitude), "lat": float(row.latitude)},
    )
    for i, row in sample.iterrows()
]
fc = ee.FeatureCollection(features)
print(f"  {fc.size().getInfo()} features in {time.time()-t0:.1f}s")

# ─── Helper: GEE extract → DataFrame ─────────────────────────────────────────

def extract(image: ee.Image, fc: ee.FeatureCollection, scale: int) -> pd.DataFrame:
    """reduceRegions with first() reducer → pandas DataFrame."""
    result = image.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=scale)
    rows = [f["properties"] for f in result.getInfo()["features"]]
    return pd.DataFrame(rows).set_index("pid")

# ─── 4. SRTM + Hansen static ─────────────────────────────────────────────────

print("\nSTEP 4a — Extract SRTM + Hansen static")
t0 = time.time()

srtm = ee.Image("USGS/SRTMGL1_003")
hansen = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")

static_img = (
    srtm.select("elevation")
    .addBands(ee.Terrain.slope(srtm).rename("slope"))
    .addBands(hansen.select("treecover2000"))
    .addBands(hansen.select("lossyear").rename("lossyear_raw"))
)
df_static = extract(static_img, fc, scale=30)
print(f"  Done in {time.time()-t0:.1f}s — columns: {list(df_static.columns)}")

# Compute per-year deforestation from lossyear_raw (value = year - 2000)
for yr in YEARS:
    col = yr - 2000  # e.g. 2017 → 17
    df_static[f"deforested_{yr}"] = (df_static["lossyear_raw"] == col).astype(float)

# Cumulative loss 2001-2021 (before target year)
df_static["cum_loss_2001_2021"] = df_static["lossyear_raw"].between(1, 21).astype(float)

# ─── 5. WorldPop annual ──────────────────────────────────────────────────────

print("\nSTEP 4b — Extract WorldPop (annual population density)")
t0 = time.time()

# COD (DRC) available 2000-2020; band name = "population"
worldpop = ee.ImageCollection("WorldPop/GP/100m/pop")
dfs_pop = {}
for yr in YEARS:
    yr_eff = min(yr, 2020)  # clamp to available range
    img = (
        worldpop
        .filter(ee.Filter.eq("country", "COD"))
        .filter(ee.Filter.eq("year", yr_eff))
        .first()
        .select("population")
        .rename(f"pop_{yr}")
    )
    df_yr = extract(img, fc, scale=100)
    if f"pop_{yr}" in df_yr.columns:
        dfs_pop[yr] = df_yr[[f"pop_{yr}"]]
        valid = dfs_pop[yr][f"pop_{yr}"].notna().sum()
        print(f"  WorldPop {yr} (using {yr_eff}): {valid}/{N_POINTS} valid")
    else:
        # GEE may rename to "first" — rename if needed
        candidates = [c for c in df_yr.columns if "pop" in c.lower() or "first" in c.lower() or "population" in c.lower()]
        if candidates:
            dfs_pop[yr] = df_yr[[candidates[0]]].rename(columns={candidates[0]: f"pop_{yr}"})
            valid = dfs_pop[yr][f"pop_{yr}"].notna().sum()
            print(f"  WorldPop {yr} (renamed from {candidates[0]}): {valid}/{N_POINTS} valid")
        else:
            print(f"  WorldPop {yr}: no matching column in {list(df_yr.columns)[:5]} — NaN")
            dfs_pop[yr] = pd.DataFrame({f"pop_{yr}": np.nan}, index=df_static.index)

df_pop = pd.concat(dfs_pop.values(), axis=1)
print(f"  Done in {time.time()-t0:.1f}s")

# ─── 6. CHIRPS annual profiles ───────────────────────────────────────────────

print("\nSTEP 4c — Extract CHIRPS annual climate profiles")
t0 = time.time()

def chirps_annual_profile(year: int) -> ee.Image:
    """Server-side annual stats from daily CHIRPS."""
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year, 12, 31)
    daily = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(start, end)
    return (
        daily.sum().rename(f"precip_total_{year}")
        .addBands(daily.mean().rename(f"precip_mean_{year}"))
        .addBands(
            daily.map(lambda img: img.lt(1).rename("dry"))
            .sum().rename(f"dry_days_{year}")
        )
        .addBands(
            daily.map(lambda img: img.gt(50).rename("ext"))
            .sum().rename(f"extreme_rain_days_{year}")
        )
    )

dfs_chirps = {}
for yr in YEARS:
    img = chirps_annual_profile(yr)
    cols = [f"precip_total_{yr}", f"precip_mean_{yr}",
            f"dry_days_{yr}", f"extreme_rain_days_{yr}"]
    dfs_chirps[yr] = extract(img, fc, scale=5000)[cols]

df_chirps = pd.concat(dfs_chirps.values(), axis=1)
print(f"  Done in {time.time()-t0:.1f}s")

# ─── 7. Merge all features ───────────────────────────────────────────────────

print("\nSTEP 5 — Merge features")

# Re-index sample with pid
sample_idx = sample.copy()
sample_idx.index.name = "pid"

df_all = (
    sample_idx[["longitude", "latitude", "deforested_2022"]]
    .join(df_static.drop(columns=["lon", "lat"], errors="ignore"))
    .join(df_pop)
    .join(df_chirps)
)

# ─── 8. Temporal feature engineering ─────────────────────────────────────────

print("STEP 6 — Temporal feature engineering")

# Population growth Δ1yr, Δ3yr
for i, yr in enumerate(YEARS[1:], 1):
    prev = YEARS[i - 1]
    df_all[f"pop_delta1_{yr}"] = df_all[f"pop_{yr}"] - df_all[f"pop_{prev}"]

if len(YEARS) >= 4:
    df_all["pop_trend_3yr"] = df_all[f"pop_{YEARS[-1]}"] - df_all[f"pop_{YEARS[-4]}"]

# Precipitation anomaly vs mean over YEARS
precip_cols = [f"precip_total_{yr}" for yr in YEARS]
df_all["precip_mean_all"] = df_all[precip_cols].mean(axis=1)
for yr in YEARS:
    df_all[f"precip_anom_{yr}"] = df_all[f"precip_total_{yr}"] - df_all["precip_mean_all"]

# Precipitation trend (last - first)
df_all["precip_trend"] = df_all[f"precip_total_{YEARS[-1]}"] - df_all[f"precip_total_{YEARS[0]}"]

# Loss momentum: deforested in recent years
df_all["loss_last2yrs"] = df_all[[f"deforested_{yr}" for yr in YEARS[-2:]]].sum(axis=1)
df_all["loss_last5yrs"] = df_all[[f"deforested_{yr}" for yr in YEARS]].sum(axis=1)

# Forest remaining after deforestation history
df_all["forest_remaining"] = (
    df_all["treecover2000"] / 100.0
    - df_all[[f"deforested_{yr}" for yr in YEARS]].sum(axis=1) / 100.0
).clip(lower=0)

print(f"  Final feature matrix: {df_all.shape}")
print(f"  NaN count: {df_all.isna().sum().sum()}")

# Save features
feats_path = OUTPUT_DIR / "mini_pipeline_features_1k.parquet"
df_all.to_parquet(feats_path)
print(f"  Saved to {feats_path}")

# ─── 9. Train XGBoost ────────────────────────────────────────────────────────

print("\nSTEP 7 — Train XGBoost")

FEATURE_COLS = [c for c in df_all.columns if c not in
                ("deforested_2022", "longitude", "latitude", "lossyear_raw",
                 "precip_mean_all")]

# Impute NaN with column median (keeps all rows)
df_model = df_all[FEATURE_COLS + ["deforested_2022"]].copy()
nan_before = df_model[FEATURE_COLS].isna().sum().sum()
for col in FEATURE_COLS:
    med = df_model[col].median()
    df_model[col] = df_model[col].fillna(med)
print(f"  {len(df_model)} rows × {len(FEATURE_COLS)} features")
print(f"  Imputed {nan_before} NaN values with column medians")

X = df_model[FEATURE_COLS].values
y = df_model["deforested_2022"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,  # balanced by construction
    eval_metric="auc",
    early_stopping_rounds=20,
    random_state=SEED,
    verbosity=0,
)

t0 = time.time()
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
print(f"  Trained in {time.time()-t0:.1f}s ({model.best_iteration} trees)")

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_proba)
print(f"\n  AUC-ROC: {auc:.3f}")
print(classification_report(y_test, y_pred, target_names=["No loss", "Deforested"]))

# ─── 10. SHAP ────────────────────────────────────────────────────────────────

print("\nSTEP 8 — SHAP analysis")
t0 = time.time()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
print(f"  SHAP computed in {time.time()-t0:.1f}s")

# Global importance
shap_abs_mean = np.abs(shap_values).mean(axis=0)
importance = pd.Series(shap_abs_mean, index=FEATURE_COLS).sort_values(ascending=False)

print("\n  Top 15 features by mean |SHAP|:")
print(importance.head(15).to_string())

# ─── 11. Plots ───────────────────────────────────────────────────────────────

print("\nSTEP 9 — Plots")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: SHAP bar chart
top15 = importance.head(15)
axes[0].barh(top15.index[::-1], top15.values[::-1], color="steelblue")
axes[0].set_xlabel("Mean |SHAP value|")
axes[0].set_title("Top 15 Features — Global SHAP Importance")
axes[0].tick_params(axis='y', labelsize=8)

# Right: AUC score text + feature count
axes[1].axis("off")
summary = (
    f"Mini Pipeline Results\n\n"
    f"Points: {len(df_model)}\n"
    f"Features: {len(FEATURE_COLS)}\n"
    f"Sources: SRTM, Hansen, WorldPop, CHIRPS\n"
    f"Years: {YEARS[0]}–{YEARS[-1]}\n\n"
    f"AUC-ROC: {auc:.3f}\n\n"
    f"Top feature: {importance.index[0]}"
)
axes[1].text(0.1, 0.5, summary, transform=axes[1].transAxes,
             fontsize=12, va="center", family="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
plot_path = OUTPUT_DIR / "mini_pipeline_shap.png"
plt.savefig(plot_path, dpi=120, bbox_inches="tight")
print(f"  Plot saved to {plot_path}")

print("\n" + "=" * 60)
print(f"MINI PIPELINE COMPLETE")
print(f"  Features: {feats_path.name}")
print(f"  SHAP plot: {plot_path.name}")
print(f"  AUC-ROC: {auc:.3f}")
print("=" * 60)
