"""Smoke test: extract SRTM + Hansen at 100 sample points via GEE.

Usage:
    conda activate deforest
    python scripts/smoke_test_gee.py
"""

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

import ee
from data.gee_utils import init_gee

LEGACY_CSV = Path(
    "/Users/guillaume/Documents/Clouds/UGA/Recherche/deforestation"
    "/src/data/tiles_250000_10N_020E_20231023.csv"
)

# --- 1. Initialize GEE ---
print("Initializing GEE...")
init_gee()
print("  OK")

# --- 2. Load sample points ---
print(f"\nLoading legacy CSV: {LEGACY_CSV}")
tiles = pd.read_csv(LEGACY_CSV, usecols=["longitude", "latitude", "lossyear_22_mean"])
print(f"  {len(tiles):,} points loaded")

# Take 100 random points for smoke test
sample = tiles.sample(100, random_state=42).reset_index(drop=True)
print(f"  Using {len(sample)} points for smoke test")

# --- 3. Upload points to GEE ---
print("\nCreating GEE FeatureCollection...")
features = []
for _, row in sample.iterrows():
    point = ee.Geometry.Point([row["longitude"], row["latitude"]])
    feat = ee.Feature(
        point,
        {
            "lon": float(row["longitude"]),
            "lat": float(row["latitude"]),
            "deforestation_2022": float(row["lossyear_22_mean"]),
        },
    )
    features.append(feat)

fc = ee.FeatureCollection(features)
print(f"  {fc.size().getInfo()} features uploaded")

# --- 4. Extract SRTM + Hansen ---
print("\nExtracting SRTM + Hansen at 100 points...")
t0 = time.time()

srtm = ee.Image("USGS/SRTMGL1_003")
hansen = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")

static_image = (
    srtm.select("elevation")
    .addBands(ee.Terrain.slope(srtm).rename("slope"))
    .addBands(hansen.select("treecover2000"))
    .addBands(hansen.select("lossyear"))
    .addBands(hansen.select("gain"))
)

result = static_image.reduceRegions(
    collection=fc,
    reducer=ee.Reducer.first(),
    scale=30,
)

# Fetch results
data = result.getInfo()
elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s")

# --- 5. Parse results ---
rows = []
for feat in data["features"]:
    props = feat["properties"]
    rows.append(props)

df = pd.DataFrame(rows)
print(f"\nResult: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print(f"\nSample values:")
print(df[["lon", "lat", "elevation", "slope", "treecover2000", "lossyear"]].head(10).to_string())

# --- 6. Sanity checks ---
print("\n=== Sanity checks ===")
assert len(df) == 100, f"Expected 100 rows, got {len(df)}"
assert df["elevation"].notna().all(), "Missing elevation values!"
assert (df["elevation"] > 0).all(), "Elevation should be > 0 in Congo Basin"
assert df["treecover2000"].notna().all(), "Missing treecover values!"

print(f"  Elevation range: {df['elevation'].min():.0f} — {df['elevation'].max():.0f} m")
print(f"  Slope range: {df['slope'].min():.1f} — {df['slope'].max():.1f}°")
print(f"  Treecover2000 range: {df['treecover2000'].min():.0f} — {df['treecover2000'].max():.0f}%")
print(f"  Lossyear: {(df['lossyear'] > 0).sum()} points with loss detected")

# --- 7. Save ---
output_path = PROJECT_DIR / "data" / "smoke_test_100pts.parquet"
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(output_path, index=False)
print(f"\nSaved to {output_path}")

print("\n✓ Smoke test PASSED — GEE extraction works!")
