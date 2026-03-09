"""Generate a high-resolution prediction map for a small demo zone.

Creates a dense 250m grid in a bbox in NE Congo (mixed forest/deforestation
frontier), extracts all GEE features, runs the trained XGBoost model, and
exports predictions.

The prediction is a TRUE FORECAST — no target data exists for 2025.

Steps:
  1. Generate ~14K points (250m spacing) in [27.0-27.3°E, 2.4-2.7°N]
  2. Extract 7 GEE sources for years 2020-2024
  3. Add spatial features (WDPA + OSM roads/settlements)
  4. Add tabular features (WGI, WDI, commodity prices)
  5. Build lag-indexed features (rebuild_features_dataset, pred_yr=2025)
  6. Load model, align features, predict
  7. Export to data/app/demo_zone_predictions.parquet

Usage:
    conda activate deforest
    python scripts/demo_zone.py                   # full pipeline
    python scripts/demo_zone.py --from-raw PATH   # skip GEE, start from raw
    python scripts/demo_zone.py --predict-only     # skip extraction, just predict
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

OUTPUT_DIR = PROJECT_DIR / "data"
APP_DIR = OUTPUT_DIR / "app"

# ── Demo zone: NE Congo (mixed forest / deforestation frontier) ──────────────
BBOX = {
    "lon_min": 27.00,
    "lon_max": 27.30,
    "lat_min": 2.40,
    "lat_max": 2.70,
}
SPACING_DEG = 0.0025  # ~250m at equator

# ── Temporal config ──────────────────────────────────────────────────────────
# Extract years 2020-2024 to get:
#   - 4 lags for pred_yr=2025 (Lag1=2024, Lag2=2023, Lag3=2022, Lag4=2021)
#   - Year 2020 for delta computations (d1 at 2021 needs 2020)
FEATURE_YEARS = list(range(2020, 2025))  # [2020, 2021, 2022, 2023, 2024]
PREDICTION_YEARS = [2025]
FEATURE_WINDOW = 4
BUFFERS_M = [150, 500, 1500, 5000]

# WorldPop only has 2020 — forward-fill to later years
WORLDPOP_AVAILABLE = [2020]

MODEL_PATH = OUTPUT_DIR / "model_20260307.json"


def generate_grid() -> pd.DataFrame:
    """Create a dense lat/lon grid within BBOX."""
    lons = np.arange(BBOX["lon_min"], BBOX["lon_max"], SPACING_DEG)
    lats = np.arange(BBOX["lat_min"], BBOX["lat_max"], SPACING_DEG)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    points = pd.DataFrame({
        "lon": lon_grid.ravel(),
        "lat": lat_grid.ravel(),
    })
    points.index.name = "pid"
    print(f"Generated {len(points):,} points ({len(lons)} x {len(lats)}) "
          f"at {SPACING_DEG*111:.0f}m spacing")
    return points


def extract_gee_features(points: pd.DataFrame) -> pd.DataFrame:
    """Extract all GEE features for the demo zone points."""
    from data.gee_utils import init_gee
    from data.gee_extraction import (
        extract_srtm,
        extract_hansen_static,
        extract_worldpop,
        extract_chirps,
        extract_era5,
        extract_viirs,
        extract_modis_fire,
        extract_hansen_buffers,
    )

    init_gee()

    df_raw = points[["lon", "lat"]].copy()

    # Static features
    t0 = time.time()
    print("\n--- SRTM (elevation, slope) ---")
    df_raw = df_raw.join(extract_srtm(points))
    print(f"  Done in {time.time()-t0:.0f}s")

    t0 = time.time()
    print("\n--- Hansen static (treecover2000, lossyear) ---")
    df_raw = df_raw.join(extract_hansen_static(points))
    print(f"  Done in {time.time()-t0:.0f}s")

    # Temporal features
    t0 = time.time()
    print(f"\n--- WorldPop (years={WORLDPOP_AVAILABLE}) ---")
    df_wp = extract_worldpop(points, years=WORLDPOP_AVAILABLE)
    # Forward-fill 2020 to 2021-2024
    for yr in FEATURE_YEARS:
        if yr not in WORLDPOP_AVAILABLE:
            df_wp[f"pop_{yr}"] = df_wp["pop_2020"]
    df_raw = df_raw.join(df_wp)
    print(f"  Done in {time.time()-t0:.0f}s (forward-filled to {FEATURE_YEARS[-1]})")

    t0 = time.time()
    print(f"\n--- CHIRPS precipitation (years={FEATURE_YEARS}) ---")
    df_raw = df_raw.join(extract_chirps(points, years=FEATURE_YEARS))
    print(f"  Done in {time.time()-t0:.0f}s")

    t0 = time.time()
    print(f"\n--- ERA5-Land (years={FEATURE_YEARS}) ---")
    df_raw = df_raw.join(extract_era5(points, years=FEATURE_YEARS))
    print(f"  Done in {time.time()-t0:.0f}s")

    t0 = time.time()
    print(f"\n--- VIIRS nighttime lights (years={FEATURE_YEARS}) ---")
    df_raw = df_raw.join(extract_viirs(points, years=FEATURE_YEARS))
    print(f"  Done in {time.time()-t0:.0f}s")

    t0 = time.time()
    print(f"\n--- MODIS fire (years={FEATURE_YEARS}) ---")
    df_raw = df_raw.join(extract_modis_fire(points, years=FEATURE_YEARS))
    print(f"  Done in {time.time()-t0:.0f}s")

    # Hansen buffers (slow — raster method)
    # Buffer years: need defo rates for years within the feature window
    # For pred_yr=2025 with window=4: lags from 2021-2024
    buffer_years = [y for y in FEATURE_YEARS if y <= 2024]
    t0 = time.time()
    print(f"\n--- Hansen buffers {BUFFERS_M} (years={buffer_years}) ---")
    df_raw = df_raw.join(extract_hansen_buffers(
        points, years=buffer_years, buffers_m=BUFFERS_M
    ))
    print(f"  Done in {time.time()-t0:.0f}s")

    print(f"\nRaw extraction complete: {df_raw.shape}")
    return df_raw


def add_spatial_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Add WDPA + OSM distance features (same logic as add_spatial_features.py)."""
    from data.gee_utils import init_gee
    from data.gee_extraction import extract_wdpa
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    from shapely.ops import unary_union
    import requests

    init_gee()

    # WDPA
    print("\n--- WDPA (protected areas) ---")
    df_wdpa = extract_wdpa(df_raw[["lon", "lat"]])
    for col in df_wdpa.columns:
        df_raw[col] = df_wdpa[col]

    # IUCN dummies (model expects these)
    for cat in ["strict", "moderate", "sustainable", "not_reported"]:
        if f"iucn_{cat}" not in df_raw.columns:
            df_raw[f"iucn_{cat}"] = 0
    if "pa_defo_rate" not in df_raw.columns:
        df_raw["pa_defo_rate"] = 0.0
    if "pa_pressure_ring" not in df_raw.columns:
        df_raw["pa_pressure_ring"] = 0.0

    # OSM roads
    print("\n--- OSM roads ---")
    TILE_CRS = "EPSG:32634"
    s, w = BBOX["lat_min"] - 0.1, BBOX["lon_min"] - 0.1
    n, e = BBOX["lat_max"] + 0.1, BBOX["lon_max"] + 0.1

    url = "https://overpass-api.de/api/interpreter"
    ql = f"""[out:json][timeout:180];way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"]({s},{w},{n},{e});out geom;"""
    try:
        resp = requests.post(url, data={"data": ql}, timeout=240)
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
        geoms = []
        for el in elements:
            if el.get("type") == "way" and "geometry" in el:
                coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
                if len(coords) >= 2:
                    geoms.append(LineString(coords))
        if geoms:
            roads = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326").to_crs(TILE_CRS)
            road_union = unary_union(roads.geometry)
            pts_gdf = gpd.GeoDataFrame(
                df_raw[["lon", "lat"]],
                geometry=[Point(x, y) for x, y in zip(df_raw.lon, df_raw.lat)],
                crs="EPSG:4326",
            ).to_crs(TILE_CRS)
            df_raw["dist_road_km"] = pts_gdf.geometry.distance(road_union) / 1000.0
            print(f"  {len(geoms)} road segments, dist computed")
        else:
            df_raw["dist_road_km"] = np.nan
            print("  No roads found, NaN")
    except Exception as exc:
        print(f"  OSM roads failed: {exc}")
        df_raw["dist_road_km"] = np.nan

    # OSM settlements
    print("\n--- OSM settlements ---")
    ql_set = f"""[out:json][timeout:60];node["place"~"^(city|town|village|hamlet)$"]({s},{w},{n},{e});out;"""
    try:
        resp = requests.post(url, data={"data": ql_set}, timeout=120)
        resp.raise_for_status()
        nodes = resp.json().get("elements", [])
        if nodes:
            sett_pts = [Point(nd["lon"], nd["lat"]) for nd in nodes]
            sett_gdf = gpd.GeoDataFrame(geometry=sett_pts, crs="EPSG:4326").to_crs(TILE_CRS)
            sett_union = unary_union(sett_gdf.geometry)
            pts_gdf_s = gpd.GeoDataFrame(
                df_raw[["lon", "lat"]],
                geometry=[Point(x, y) for x, y in zip(df_raw.lon, df_raw.lat)],
                crs="EPSG:4326",
            ).to_crs(TILE_CRS)
            df_raw["dist_settlement_km"] = pts_gdf_s.geometry.distance(sett_union) / 1000.0
            print(f"  {len(nodes)} settlements, dist computed")
        else:
            df_raw["dist_settlement_km"] = np.nan
            print("  No settlements found, NaN")
    except Exception as exc:
        print(f"  OSM settlements failed: {exc}")
        df_raw["dist_settlement_km"] = np.nan

    return df_raw


def add_tabular_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Add country dummies and country-level indicators."""
    import geopandas as gpd
    from shapely.geometry import Point

    # Country assignment
    print("\n--- Country assignment ---")
    ne_path = PROJECT_DIR / "data" / "boundaries" / "ne_110m_countries.gpkg"
    if ne_path.exists():
        world = gpd.read_file(ne_path)
        pts_gdf = gpd.GeoDataFrame(
            df_raw[["lon", "lat"]],
            geometry=[Point(x, y) for x, y in zip(df_raw.lon, df_raw.lat)],
            crs="EPSG:4326",
        )
        joined = gpd.sjoin(pts_gdf, world[["iso_a3", "geometry"]], how="left", predicate="within")
        df_raw["country_iso3"] = joined["iso_a3"].values
        # Fix South Sudan
        df_raw["country_iso3"] = df_raw["country_iso3"].replace({"SDS": "SSD"})
    else:
        # Fallback: the demo zone is in DRC
        df_raw["country_iso3"] = "COD"

    # Country dummies (model expects these)
    for country in ["CAF", "COD", "SDN", "SDS", "SSD", "TCD", "UGA"]:
        df_raw[f"country_{country}"] = (df_raw["country_iso3"] == country).astype(float)

    # Commodity prices (global, not per country)
    _PRICES = {
        "price_cocoa": {2020: 2373, 2021: 2428, 2022: 2403, 2023: 3399, 2024: 7000},
        "price_palmoil": {2020: 752, 2021: 1131, 2022: 1282, 2023: 969, 2024: 1000},
        "price_gold": {2020: 1770, 2021: 1799, 2022: 1800, 2023: 1943, 2024: 2300},
    }
    for var, prices in _PRICES.items():
        for yr in FEATURE_YEARS:
            df_raw[f"{var}_{yr}"] = prices.get(yr, prices[max(prices.keys())])

    # WGI/WDI: simplified — use DRC values (the demo zone is entirely in DRC)
    # These have minimal impact on predictions (economy group = 0.5 AUC)
    # Using approximate values for DRC
    _WGI_COD = {
        "control_corruption": -1.5, "gov_effectiveness": -1.8,
        "political_stability": -2.3, "rule_of_law": -1.7,
        "regulatory_quality": -1.5, "voice_accountability": -1.3,
    }
    _WDI_COD = {
        "gdp_pc": 580, "pop_growth": 3.2, "urbanization": 46,
        "inflation": 9.0, "fdi": 2.0, "military_exp": 0.7, "fertility": 6.0,
    }
    for var, val in {**_WGI_COD, **_WDI_COD}.items():
        for yr in FEATURE_YEARS:
            df_raw[f"{var}_{yr}"] = val

    print(f"  Tabular features added, shape: {df_raw.shape}")
    return df_raw


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Build lag-indexed features using rebuild_features_dataset."""
    from data.gee_extraction import rebuild_features_dataset
    from train_xgboost import add_window_summaries

    print(f"\nBuilding features (pred_yr={PREDICTION_YEARS}, window={FEATURE_WINDOW})...")
    df = rebuild_features_dataset(
        df_raw,
        years=FEATURE_YEARS,
        prediction_years=PREDICTION_YEARS,
        feature_window=FEATURE_WINDOW,
    )
    df = add_window_summaries(df)
    df["split"] = "demo"
    print(f"  Features shape: {df.shape}")
    return df


def predict(df_features: pd.DataFrame) -> pd.DataFrame:
    """Load model and run inference."""
    import json
    import xgboost as xgb

    print(f"\nLoading model from {MODEL_PATH.name}...")
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    # Get expected feature columns
    with open(APP_DIR / "model_info.json") as f:
        model_info = json.load(f)
    expected_cols = model_info["feature_cols"]
    print(f"  Model expects {len(expected_cols)} features")

    # Align features: add missing columns as NaN, drop extra columns
    available = set(df_features.columns)
    missing = [c for c in expected_cols if c not in available]
    extra = [c for c in available if c in expected_cols]
    if missing:
        print(f"  Missing features ({len(missing)}): {missing[:10]}{'...' if len(missing)>10 else ''}")
        for c in missing:
            df_features[c] = np.nan
    print(f"  Available: {len(extra)}/{len(expected_cols)} features")

    X = df_features[expected_cols].values
    y_proba = model.predict_proba(X)[:, 1]

    out = df_features[["lon", "lat"]].copy()
    out["proba"] = y_proba
    out["risk_pct"] = (y_proba * 100).round(2)

    print(f"\nPrediction stats:")
    print(f"  N points:  {len(out):,}")
    print(f"  Mean risk: {y_proba.mean()*100:.2f}%")
    print(f"  Max risk:  {y_proba.max()*100:.2f}%")
    print(f"  >1% risk:  {(y_proba>0.01).sum():,} ({(y_proba>0.01).mean()*100:.1f}%)")
    print(f"  >5% risk:  {(y_proba>0.05).sum():,} ({(y_proba>0.05).mean()*100:.1f}%)")

    return out


def main():
    parser = argparse.ArgumentParser(description="Demo zone prediction pipeline")
    parser.add_argument("--from-raw", type=str, help="Skip GEE, load raw parquet")
    parser.add_argument("--predict-only", action="store_true",
                        help="Skip extraction, load features parquet")
    args = parser.parse_args()

    raw_path = OUTPUT_DIR / "demo_zone_raw.parquet"
    feat_path = OUTPUT_DIR / "demo_zone_features.parquet"

    if args.predict_only:
        print("Loading pre-built features...")
        df_features = pd.read_parquet(feat_path)
    else:
        if args.from_raw:
            print(f"Loading raw from {args.from_raw}...")
            df_raw = pd.read_parquet(args.from_raw)
        else:
            # Full extraction
            points = generate_grid()
            df_raw = extract_gee_features(points)
            df_raw = add_spatial_features(df_raw)
            df_raw = add_tabular_features(df_raw)

            # Save raw checkpoint
            df_raw.to_parquet(raw_path)
            print(f"\nRaw checkpoint saved: {raw_path.name} ({raw_path.stat().st_size/1e6:.0f} MB)")

        # Feature engineering
        df_features = build_features(df_raw)
        df_features.to_parquet(feat_path, index=False)
        print(f"Features saved: {feat_path.name}")

    # Inference
    predictions = predict(df_features)

    # Export
    out_path = APP_DIR / "demo_zone_predictions.parquet"
    predictions.to_parquet(out_path, index=False)
    print(f"\nExported: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    print(f"Done! {len(predictions):,} predictions for {PREDICTION_YEARS[0]}")


if __name__ == "__main__":
    main()
