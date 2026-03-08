"""Add static spatial features to an existing raw parquet checkpoint.

Features added (time-invariant, passed through the sliding window as static):
  - in_protected, dist_protected_km   (WDPA via GEE)
  - dist_road_km                       (OSM major roads via Overpass API)
  - dist_settlement_km                 (OSM settlements via Overpass API)

Usage:
    conda activate deforest
    python scripts/add_spatial_features.py --raw data/raw_250k_20260228.parquet
    python scripts/add_spatial_features.py --raw data/raw_250k_20260228.parquet --skip-osm
"""

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString, Point

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data.gee_utils import init_gee
from data.gee_extraction import extract_wdpa, rebuild_features_dataset

YEARS = [2016, 2017, 2018, 2019, 2020, 2021]
PREDICTION_YEARS = [2019, 2020, 2021, 2022]

# UTM Zone 34N — covers most of the Congo Basin tile (0-10°N, 20-30°E)
TILE_CRS = "EPSG:32634"


# ─── OSM helpers ──────────────────────────────────────────────────────────────


def _overpass_roads(
    s: float, w: float, n: float, e: float, timeout: int = 180
) -> gpd.GeoDataFrame:
    """Download major OSM roads (motorway → tertiary) via Overpass API.

    Returns GeoDataFrame of LineStrings projected to TILE_CRS.
    Empty GeoDataFrame if the query fails or returns no data.
    """
    url = "https://overpass-api.de/api/interpreter"
    ql = f"""
    [out:json][timeout:{timeout}];
    way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"]({s},{w},{n},{e});
    out geom;
    """
    print(f"  Querying OSM roads ({s:.1f},{w:.1f})→({n:.1f},{e:.1f})…")
    resp = requests.post(url, data={"data": ql}, timeout=timeout + 60)
    resp.raise_for_status()
    elements = resp.json().get("elements", [])

    geoms = []
    for el in elements:
        if el.get("type") == "way" and "geometry" in el:
            coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
            if len(coords) >= 2:
                geoms.append(LineString(coords))

    if not geoms:
        print("  ⚠ No road segments returned.")
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326")).to_crs(TILE_CRS)

    gdf = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")
    print(f"  {len(gdf):,} road segments downloaded.")
    return gdf.to_crs(TILE_CRS)


def _overpass_settlements(
    s: float, w: float, n: float, e: float, timeout: int = 60
) -> gpd.GeoDataFrame:
    """Download OSM settlement nodes (city/town/village/hamlet) via Overpass.

    Returns GeoDataFrame of Points projected to TILE_CRS.
    """
    url = "https://overpass-api.de/api/interpreter"
    ql = f"""
    [out:json][timeout:{timeout}];
    node["place"~"^(city|town|village|hamlet)$"]({s},{w},{n},{e});
    out body;
    """
    print(f"  Querying OSM settlements ({s:.1f},{w:.1f})→({n:.1f},{e:.1f})…")
    resp = requests.post(url, data={"data": ql}, timeout=timeout + 30)
    resp.raise_for_status()
    elements = resp.json().get("elements", [])

    records = [
        {"geometry": Point(el["lon"], el["lat"])}
        for el in elements
        if el.get("type") == "node"
    ]
    if not records:
        print("  ⚠ No settlement nodes returned.")
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326")).to_crs(TILE_CRS)

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    print(f"  {len(gdf):,} settlement nodes downloaded.")
    return gdf.to_crs(TILE_CRS)


def _dist_to_features(
    points_gdf: gpd.GeoDataFrame,
    ref_gdf: gpd.GeoDataFrame,
    col_name: str,
) -> pd.Series:
    """Compute distance (km) from each point to its nearest feature in ref_gdf.

    Uses geopandas.sjoin_nearest on projected coordinates (metres).
    Returns NaN series if ref_gdf is empty.
    """
    if len(ref_gdf) == 0:
        return pd.Series(np.nan, index=points_gdf.index, name=col_name)

    joined = gpd.sjoin_nearest(
        points_gdf[["geometry"]],
        ref_gdf[["geometry"]].reset_index(drop=True),
        how="left",
        distance_col="_dist_m",
    )
    # Keep first match if multiple (shouldn't happen for left join)
    joined = joined[~joined.index.duplicated(keep="first")]
    s = (joined["_dist_m"] / 1000.0).rename(col_name)
    return s


# ─── Main ─────────────────────────────────────────────────────────────────────


def main(raw_path: Path, skip_osm: bool) -> None:
    print("=" * 60)
    print("ADD SPATIAL FEATURES — WDPA + OSM Roads + Settlements")
    print("=" * 60)

    # ── 1. Load raw checkpoint ────────────────────────────────────────────────
    print(f"\nLoading: {raw_path.name}")
    df_raw = pd.read_parquet(raw_path)
    n_pts = len(df_raw)
    print(f"  {n_pts:,} points, {df_raw.shape[1]} columns")

    points = df_raw[["lon", "lat"]].copy()
    points.index.name = "pid"

    s = float(points["lat"].min()) - 0.05
    w = float(points["lon"].min()) - 0.05
    n = float(points["lat"].max()) + 0.05
    e = float(points["lon"].max()) + 0.05

    # ── 2. WDPA via GEE ───────────────────────────────────────────────────────
    print("\nInitializing GEE…")
    init_gee()
    print("  OK")

    print("\n[1] WDPA protected areas…")
    t0 = time.time()
    df_wdpa = extract_wdpa(points)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s — {df_wdpa.shape[1]} columns")
    if "in_protected" in df_wdpa.columns:
        pct = df_wdpa["in_protected"].mean() * 100
        med_dist = df_wdpa["dist_protected_km"].median() if "dist_protected_km" in df_wdpa.columns else float("nan")
        print(f"    in_protected: {pct:.1f}% inside a PA")
        print(f"    dist_protected_km: median {med_dist:.1f} km")

    # ── 3. OSM Roads ──────────────────────────────────────────────────────────
    df_road_dist = None
    df_settle_dist = None

    if not skip_osm:
        # Build projected point GeoDataFrame once
        points_gdf = gpd.GeoDataFrame(
            points,
            geometry=gpd.points_from_xy(points["lon"], points["lat"]),
            crs="EPSG:4326",
        ).to_crs(TILE_CRS)

        print("\n[2] OSM roads (dist_road_km)…")
        try:
            t0 = time.time()
            roads_gdf = _overpass_roads(s, w, n, e)
            if len(roads_gdf) > 0:
                df_road_dist = _dist_to_features(points_gdf, roads_gdf, "dist_road_km")
                print(f"  Distances in {time.time()-t0:.0f}s | median {df_road_dist.median():.2f} km")
        except Exception as exc:
            print(f"  ⚠ Roads query failed: {exc}")

        # ── 4. OSM Settlements ─────────────────────────────────────────────────
        print("\n[3] OSM settlements (dist_settlement_km)…")
        try:
            t0 = time.time()
            settle_gdf = _overpass_settlements(s, w, n, e)
            if len(settle_gdf) > 0:
                df_settle_dist = _dist_to_features(points_gdf, settle_gdf, "dist_settlement_km")
                print(f"  Distances in {time.time()-t0:.0f}s | median {df_settle_dist.median():.2f} km")
        except Exception as exc:
            print(f"  ⚠ Settlements query failed: {exc}")
    else:
        print("\n[2] Skipping OSM (--skip-osm)")

    # ── 5. Patch raw parquet ──────────────────────────────────────────────────
    print("\nPatching raw parquet…")
    new_cols = list(df_wdpa.columns)
    if df_road_dist is not None:
        new_cols.append("dist_road_km")
    if df_settle_dist is not None:
        new_cols.append("dist_settlement_km")

    df_raw_patched = df_raw.drop(columns=[c for c in new_cols if c in df_raw.columns])
    df_raw_patched = df_raw_patched.join(df_wdpa)
    if df_road_dist is not None:
        df_raw_patched = df_raw_patched.join(df_road_dist)
    if df_settle_dist is not None:
        df_raw_patched = df_raw_patched.join(df_settle_dist)
    df_raw_patched.to_parquet(raw_path)
    print(f"  Saved: {raw_path.name} ({df_raw_patched.shape[1]} columns)")

    # ── 6. Rebuild features dataset ───────────────────────────────────────────
    print("\nRebuilding features dataset…")
    df_dataset = rebuild_features_dataset(df_raw_patched, YEARS, PREDICTION_YEARS)
    print(f"  Dataset shape: {df_dataset.shape}")

    for split in ["train", "val", "test"]:
        sub = df_dataset[df_dataset["split"] == split]
        pos = sub["target"].mean() * 100 if len(sub) > 0 else 0.0
        print(f"  {split:5s}: {len(sub):>7,} rows — {pos:.1f}% deforested")

    stem = raw_path.stem.replace("raw_", "features_")
    out_path = raw_path.parent / f"{stem}.parquet"
    df_dataset.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path.name}")

    print("\n" + "=" * 60)
    print("SPATIAL FEATURES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add WDPA + OSM features to raw parquet")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--skip-osm", action="store_true", help="Skip Overpass queries")
    args = parser.parse_args()
    main(args.raw, args.skip_osm)
