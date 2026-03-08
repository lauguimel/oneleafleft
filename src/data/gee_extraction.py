"""GEE extraction utilities for the deforestation prediction pipeline.

Design principles:
- All temporal aggregation happens server-side in GEE (minimize EECU)
- Points are batched in chunks of BATCH_SIZE (GEE limit ~5000 features)
- Each function returns a pd.DataFrame indexed by point ID
- NaN = no data at that location for that source/year
"""

from __future__ import annotations

import logging
import re
import time
from typing import Callable

import ee
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BATCH_SIZE = 4000  # safe limit for reduceRegions


# ─── Core helpers ─────────────────────────────────────────────────────────────


def _fc_from_df(df: pd.DataFrame, pid_col: str = "pid") -> ee.FeatureCollection:
    """Build an ee.FeatureCollection from a DataFrame with lon/lat columns."""
    features = [
        ee.Feature(
            ee.Geometry.Point([float(row.lon), float(row.lat)]),
            {pid_col: int(idx)},
        )
        for idx, row in df.iterrows()
    ]
    return ee.FeatureCollection(features)


def _extract_batch(
    image: ee.Image,
    fc: ee.FeatureCollection,
    scale: int,
    pid_col: str = "pid",
) -> pd.DataFrame:
    """reduceRegions(first) on a single batch → DataFrame indexed by pid."""
    result = image.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.first(),
        scale=scale,
    )
    rows = [f["properties"] for f in result.getInfo()["features"]]
    df = pd.DataFrame(rows)
    if pid_col in df.columns:
        df = df.set_index(pid_col)
    return df


def extract_image(
    image: ee.Image,
    points: pd.DataFrame,
    scale: int,
    pid_col: str = "pid",
    batch_size: int = BATCH_SIZE,
    verbose: bool = True,
    max_workers: int = 10,
) -> pd.DataFrame:
    """Extract an ee.Image at all points, batching automatically.

    Args:
        image: Single ee.Image with one or more bands already named.
        points: DataFrame with columns [lon, lat], indexed by pid.
        scale: Spatial resolution in metres.
        batch_size: Max features per GEE call.
        verbose: Print progress.
        max_workers: Number of concurrent GEE requests (default 10).

    Returns:
        DataFrame indexed by pid with one column per image band.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n = len(points)
    n_batches = (n + batch_size - 1) // batch_size

    # Prepare all batches as FeatureCollections
    batch_fcs = []
    for i in range(n_batches):
        batch = points.iloc[i * batch_size : (i + 1) * batch_size]
        fc = _fc_from_df(batch, pid_col=pid_col)
        batch_fcs.append((i, fc))

    # Execute in parallel
    results: list[pd.DataFrame | None] = [None] * n_batches
    done_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_extract_batch, image, fc, scale, pid_col): idx
            for idx, fc in batch_fcs
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            done_count += 1
            if verbose and n_batches > 1 and done_count % 10 == 0:
                print(f"    batch {done_count}/{n_batches}", flush=True)

    if verbose and n_batches > 1:
        print(f"    batch {n_batches}/{n_batches}", flush=True)

    return pd.concat(results).sort_index()


# ─── Static sources ───────────────────────────────────────────────────────────


def extract_srtm(points: pd.DataFrame, **kw) -> pd.DataFrame:
    """SRTM elevation (m) and slope (°). Scale: 30m."""
    srtm = ee.Image("USGS/SRTMGL1_003")
    img = srtm.select("elevation").addBands(
        ee.Terrain.slope(srtm).rename("slope")
    )
    return extract_image(img, points, scale=30, **kw)


def extract_hansen_static(points: pd.DataFrame, **kw) -> pd.DataFrame:
    """Hansen GFC v1.12: treecover2000 (%) and lossyear (1-24 = 2001-2024).

    lossyear=0 means no loss detected.
    """
    hansen = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    img = hansen.select(["treecover2000", "lossyear"])
    return extract_image(img, points, scale=30, **kw)


# ─── WorldPop annual (multi-country mosaic, 2000-2020) ───────────────────────


def extract_worldpop(
    points: pd.DataFrame,
    years: list[int],
    **kw,
) -> pd.DataFrame:
    """WorldPop population density (persons/pixel) for requested years.

    Uses filterBounds on the points bounding box to automatically include all
    countries in the study area (e.g. COD, COG, CAF, CMR, GAB for tile 10N_020E),
    then mosaics country images. This avoids the ~50% NaN rate from COD-only filter.

    Data available: 2000-2020. Years beyond 2020 are clamped to 2020.
    """
    worldpop = ee.ImageCollection("WorldPop/GP/100m/pop")

    # Bounding box of study area for spatial filter
    lon_min = float(points["lon"].min())
    lon_max = float(points["lon"].max())
    lat_min = float(points["lat"].min())
    lat_max = float(points["lat"].max())
    aoi = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

    dfs = {}
    for yr in years:
        yr_eff = min(yr, 2020)
        img = (
            worldpop
            .filterBounds(aoi)
            .filter(ee.Filter.eq("year", yr_eff))
            .select("population")
            .mosaic()
            .rename(f"pop_{yr}")
        )
        df = extract_image(img, points, scale=100, **kw)
        if f"pop_{yr}" not in df.columns:
            candidates = [c for c in df.columns if "pop" in c.lower()
                          or "first" in c.lower() or "population" in c.lower()]
            if candidates:
                df = df.rename(columns={candidates[0]: f"pop_{yr}"})
        dfs[yr] = df[[f"pop_{yr}"]] if f"pop_{yr}" in df.columns else \
            pd.DataFrame({f"pop_{yr}": np.nan}, index=points.index)

    return pd.concat(dfs.values(), axis=1)


# ─── CHIRPS daily → annual profiles ──────────────────────────────────────────


def _chirps_annual_image(year: int) -> ee.Image:
    """Server-side annual CHIRPS stats from daily data."""
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year, 12, 31)
    daily = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(start, end)
    return (
        daily.sum().rename(f"precip_total_{year}")
        .addBands(daily.mean().rename(f"precip_mean_{year}"))
        .addBands(daily.max().rename(f"precip_max_{year}"))
        .addBands(
            daily.map(lambda img: img.lt(1).rename("dry"))
            .sum().rename(f"dry_days_{year}")
        )
        .addBands(
            daily.map(lambda img: img.gt(50).rename("ext"))
            .sum().rename(f"extreme_rain_days_{year}")
        )
    )


def extract_chirps(
    points: pd.DataFrame,
    years: list[int],
    **kw,
) -> pd.DataFrame:
    """CHIRPS annual profiles (5 stats per year) at 5km scale."""
    dfs = []
    for yr in years:
        img = _chirps_annual_image(yr)
        cols = [f"precip_total_{yr}", f"precip_mean_{yr}", f"precip_max_{yr}",
                f"dry_days_{yr}", f"extreme_rain_days_{yr}"]
        df = extract_image(img, points, scale=5000, **kw)
        available = [c for c in cols if c in df.columns]
        dfs.append(df[available])
    return pd.concat(dfs, axis=1)


# ─── ERA5-Land annual profiles (temperature, soil moisture) ──────────────────


def _era5_annual_image(year: int) -> ee.Image:
    """Server-side annual ERA5-Land stats from daily aggregates.

    Bands extracted:
        temperature_2m_{year}      — mean annual 2m air temperature (K)
        temperature_2m_max_{year}  — mean of daily max temperature (K)
        hot_days_{year}            — days with max temp > 308.15 K (35°C)
        sm_surface_{year}          — mean surface soil moisture (m³/m³)
        sm_anom_proxy_{year}       — std of daily soil moisture (inter-annual volatility)
        et_{year}                  — mean daily potential evapotranspiration (m/day)
    """
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year, 12, 31)
    daily = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(start, end)

    temp = daily.select("temperature_2m").mean().rename(f"temperature_2m_{year}")
    temp_max = daily.select("temperature_2m_max").mean().rename(f"temperature_2m_max_{year}")
    hot_days = (
        daily.select("temperature_2m_max")
        .map(lambda img: img.gt(308.15).rename("hot"))
        .sum()
        .rename(f"hot_days_{year}")
    )
    sm = daily.select("volumetric_soil_water_layer_1").mean().rename(f"sm_surface_{year}")
    sm_std = daily.select("volumetric_soil_water_layer_1").reduce(ee.Reducer.stdDev()).rename(f"sm_std_{year}")
    et = daily.select("potential_evaporation_sum").mean().rename(f"et_{year}")

    return (
        temp
        .addBands(temp_max)
        .addBands(hot_days)
        .addBands(sm)
        .addBands(sm_std)
        .addBands(et)
    )


def extract_era5(
    points: pd.DataFrame,
    years: list[int],
    **kw,
) -> pd.DataFrame:
    """ERA5-Land annual profiles (6 stats/year) at 9km scale."""
    dfs = []
    for yr in years:
        img = _era5_annual_image(yr)
        cols = [
            f"temperature_2m_{yr}", f"temperature_2m_max_{yr}", f"hot_days_{yr}",
            f"sm_surface_{yr}", f"sm_std_{yr}", f"et_{yr}",
        ]
        df = extract_image(img, points, scale=9000, **kw)
        available = [c for c in cols if c in df.columns]
        dfs.append(df[available])
    return pd.concat(dfs, axis=1)


# ─── VIIRS nighttime lights monthly → annual profiles ─────────────────────────


def _viirs_annual_image(year: int) -> ee.Image:
    """Server-side annual VIIRS nighttime lights stats from monthly composites.

    Bands:
        ntl_mean_{year}   — mean annual radiance (nW/cm²/sr)
        ntl_max_{year}    — max monthly radiance (peak activity)
        ntl_cv_{year}     — coefficient of variation (seasonal variability)
    """
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year, 12, 31)
    monthly = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterDate(start, end)
        .select("avg_rad")
        .map(lambda img: img.max(ee.Image(0)))  # clamp negative values to 0
    )
    mean = monthly.mean().rename(f"ntl_mean_{year}")
    maxv = monthly.max().rename(f"ntl_max_{year}")
    std  = monthly.reduce(ee.Reducer.stdDev()).rename(f"ntl_std_{year}")
    # CV = std / mean (seasonal variability relative to level)
    cv = std.divide(mean.max(ee.Image(0.001))).rename(f"ntl_cv_{year}")

    return mean.addBands(maxv).addBands(std).addBands(cv)


def extract_viirs(
    points: pd.DataFrame,
    years: list[int],
    **kw,
) -> pd.DataFrame:
    """VIIRS nighttime lights annual profiles (4 stats/year) at 750m scale."""
    dfs = []
    for yr in years:
        img = _viirs_annual_image(yr)
        cols = [f"ntl_mean_{yr}", f"ntl_max_{yr}", f"ntl_std_{yr}", f"ntl_cv_{yr}"]
        df = extract_image(img, points, scale=750, **kw)
        available = [c for c in cols if c in df.columns]
        dfs.append(df[available])
    return pd.concat(dfs, axis=1)


# ─── MODIS fire annual profiles ───────────────────────────────────────────────


def _modis_fire_annual_image(year: int) -> ee.Image:
    """Server-side annual MODIS Terra fire stats.

    Bands:
        fire_days_{year}    — number of days with active fire detected
        fire_max_{year}     — max fire radiative power in a single day (MW)
        fire_season_{year}  — peak month of fire activity (1-12)
    """
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year, 12, 31)

    daily = ee.ImageCollection("MODIS/061/MOD14A1").filterDate(start, end)

    # Fire mask: FireMask band ≥ 7 = high-confidence fire
    fire_binary = daily.map(
        lambda img: img.select("FireMask").gte(7).rename("fire")
    )
    fire_days = fire_binary.sum().rename(f"fire_days_{year}")
    fire_max = (
        daily.select("MaxFRP")
        .max()
        .rename(f"fire_max_{year}")
    )

    return fire_days.addBands(fire_max)


def extract_modis_fire(
    points: pd.DataFrame,
    years: list[int],
    **kw,
) -> pd.DataFrame:
    """MODIS Terra fire annual profiles (2 stats/year) at 1km scale."""
    dfs = []
    for yr in years:
        img = _modis_fire_annual_image(yr)
        cols = [f"fire_days_{yr}", f"fire_max_{yr}"]
        df = extract_image(img, points, scale=1000, **kw)
        available = [c for c in cols if c in df.columns]
        dfs.append(df[available])
    return pd.concat(dfs, axis=1)


# ─── WDPA protected areas ─────────────────────────────────────────────────────


def extract_wdpa(points: pd.DataFrame, **kw) -> pd.DataFrame:
    """WDPA protected areas: binary membership + distance to nearest PA (km).

    Uses the WCMC/WDPA/current/polygons FeatureCollection from GEE.

    Bands:
        in_protected      — 1 if inside a WDPA polygon, 0 otherwise
        dist_protected_km — distance to nearest PA boundary (km), capped at 200 km
                            (0 if inside a PA)
    """
    wdpa = ee.FeatureCollection("WCMC/WDPA/current/polygons")

    # Binary raster: 1 inside protected area, 0 outside
    pa_mask = (
        ee.Image(0).byte()
        .paint(featureCollection=wdpa, color=1)
        .rename("in_protected")
    )

    # Distance to nearest PA pixel (up to 200 km; inside PA → 0)
    dist_km = (
        pa_mask
        .distance(ee.Kernel.euclidean(200000, "meters"))
        .divide(1000)
        .rename("dist_protected_km")
    )
    dist_km = dist_km.where(pa_mask, ee.Image(0.0))

    return extract_image(pa_mask.addBands(dist_km), points, scale=1000, **kw)


# ─── Multi-buffer spatial aggregation ────────────────────────────────────────


def extract_hansen_buffers(
    points: pd.DataFrame,
    years: list[int],
    buffers_m: list[int] | None = None,
    tile_deg: float = 2.0,
    raster_dir: str | None = None,
    **kw,
) -> pd.DataFrame:
    """Deforestation rate within spatial buffers around each point.

    Downloads the buffered deforestation image as tiled GeoTIFFs via
    ee.data.computePixels, then samples at point locations with rasterio.
    Much faster than the batch reduceRegions approach (~30 min vs 48h).

    Args:
        points: DataFrame with lon/lat, indexed by pid.
        years: List of years to compute deforestation rates for.
        buffers_m: Buffer radii in metres (default: [500, 5000]).
        tile_deg: Tile size in degrees for download chunking.
        raster_dir: If set, save GeoTIFFs to this directory.

    Returns:
        DataFrame with columns like defo_rate_500m_2019, defo_rate_5000m_2020.
    """
    import io
    from pathlib import Path

    import rasterio
    from rasterio.transform import from_bounds

    if buffers_m is None:
        buffers_m = [500, 5000]

    if raster_dir is not None:
        raster_dir = Path(raster_dir)
        raster_dir.mkdir(parents=True, exist_ok=True)

    hansen = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")

    # Scale per radius — coarser for larger buffers, no precision loss
    def _scale_m(r: int) -> int:
        if r <= 150:   return 100
        if r <= 500:   return 200
        if r <= 1500:  return 300
        return 500

    # Bounding box with padding for largest buffer
    pad = max(buffers_m) / 111320 + 0.05
    lon_min = float(points.lon.min()) - pad
    lon_max = float(points.lon.max()) + pad
    lat_min = float(points.lat.min()) - pad
    lat_max = float(points.lat.max()) + pad

    all_results = {}

    for radius in buffers_m:
        scale_m = _scale_m(radius)
        scale_deg = scale_m / 111320.0

        # Build multi-band image: one band per year, same kernel
        band_names = []
        bands = []
        for yr in years:
            name = f"defo_rate_{radius}m_{yr}"
            band_names.append(name)
            yr_code = yr - 2000
            # unmask(0): non-forest areas → 0 (not NaN). No forest = no deforestation.
            loss_yr = hansen.select("lossyear").unmask(0).eq(yr_code)
            buffered = loss_yr.reduceNeighborhood(
                reducer=ee.Reducer.mean(),
                kernel=ee.Kernel.circle(radius, "meters"),
            ).rename(name)
            bands.append(buffered)

        multi = ee.Image.cat(bands).toFloat()

        # Auto-adapt tile_deg so tiles stay within GEE's 50MB computePixels limit
        # GeoTIFF actual overhead ≈ 5.5 bytes/pixel/band (float32 + headers)
        max_side_px = int(np.sqrt(48_000_000 / (len(years) * 5.5)))
        max_tile_deg = max_side_px * scale_deg
        eff_tile_deg = min(tile_deg, max_tile_deg)

        # Tile grid
        n_tx = max(1, int(np.ceil((lon_max - lon_min) / eff_tile_deg)))
        n_ty = max(1, int(np.ceil((lat_max - lat_min) / eff_tile_deg)))

        # Assign points to tiles
        tx_idx = np.clip(
            ((points.lon.values - lon_min) / eff_tile_deg).astype(int), 0, n_tx - 1
        )
        ty_idx = np.clip(
            ((points.lat.values - lat_min) / eff_tile_deg).astype(int), 0, n_ty - 1
        )
        tile_keys = ty_idx * n_tx + tx_idx
        active_tiles = sorted(set(tile_keys))

        print(f"\n  radius={radius}m  scale={scale_m}m  "
              f"{n_tx}×{n_ty} grid, {len(active_tiles)} tiles with points",
              flush=True)

        # Initialize result arrays
        for name in band_names:
            all_results[name] = np.full(len(points), np.nan, dtype=np.float32)

        for done, tk in enumerate(active_tiles, 1):
            ty, tx = divmod(tk, n_tx)

            t_lon_min = lon_min + tx * eff_tile_deg
            t_lon_max = min(t_lon_min + eff_tile_deg, lon_max)
            t_lat_min = lat_min + ty * eff_tile_deg
            t_lat_max = min(t_lat_min + eff_tile_deg, lat_max)

            t_w = max(1, int(np.ceil((t_lon_max - t_lon_min) / scale_deg)))
            t_h = max(1, int(np.ceil((t_lat_max - t_lat_min) / scale_deg)))

            mask = tile_keys == tk
            n_pts = mask.sum()
            print(f"    tile {done}/{len(active_tiles)} "
                  f"({t_w}×{t_h}px, {n_pts:,} pts)",
                  end=" ", flush=True)

            try:
                raw = ee.data.computePixels({
                    'expression': multi,
                    'fileFormat': 'GEO_TIFF',
                    'grid': {
                        'dimensions': {'width': t_w, 'height': t_h},
                        'affineTransform': {
                            'scaleX': scale_deg,
                            'shearX': 0,
                            'translateX': t_lon_min,
                            'shearY': 0,
                            'scaleY': -scale_deg,
                            'translateY': t_lat_max,
                        },
                        'crsCode': 'EPSG:4326',
                    },
                })

                with rasterio.open(io.BytesIO(raw)) as src:
                    tile_data = src.read().astype(np.float32)  # (n_bands, h, w)
                    tile_data[~np.isfinite(tile_data)] = np.nan  # nodata → NaN
                    tile_tf = src.transform

                pt_lons = points.lon.values[mask]
                pt_lats = points.lat.values[mask]
                rows, cols = rasterio.transform.rowcol(tile_tf, pt_lons, pt_lats)
                rows = np.clip(np.array(rows), 0, t_h - 1)
                cols = np.clip(np.array(cols), 0, t_w - 1)

                for i, name in enumerate(band_names):
                    all_results[name][mask] = tile_data[i][rows, cols]

                print("ok", flush=True)

            except Exception as e:
                print(f"FAIL: {e}", flush=True)

        sampled = int(np.sum(~np.isnan(all_results[band_names[0]])))
        print(f"  Sampled: {sampled:,}/{len(points):,} points", flush=True)

        # Optionally save raster
        if raster_dir is not None:
            full_w = max(1, int(np.ceil((lon_max - lon_min) / scale_deg)))
            full_h = max(1, int(np.ceil((lat_max - lat_min) / scale_deg)))
            tf = from_bounds(lon_min, lat_min, lon_max, lat_max, full_w, full_h)
            tiff_path = raster_dir / f"defo_rate_{radius}m.tif"
            # Reconstruct full raster from sampled values is not worth it;
            # save a lightweight metadata file instead
            print(f"  (raster save skipped — values sampled directly)", flush=True)

    return pd.DataFrame(all_results, index=points.index)


# ─── Feature engineering ──────────────────────────────────────────────────────


def build_temporal_features(
    df: pd.DataFrame,
    base_var: str,
    years: list[int],
) -> pd.DataFrame:
    """Add Δ1yr, Δ3yr, trend, anomaly columns for a variable measured annually.

    Expects columns {base_var}_{yr} in df.

    Returns df with additional columns added in-place (copy).
    """
    cols = [f"{base_var}_{yr}" for yr in years if f"{base_var}_{yr}" in df.columns]
    if len(cols) < 2:
        return df

    yr_array = [yr for yr in years if f"{base_var}_{yr}" in df.columns]

    # Build new columns as a dict, then concat once to avoid DataFrame fragmentation
    new_cols: dict[str, pd.Series] = {}

    mean_series = df[cols].mean(axis=1)
    new_cols[f"{base_var}_mean"] = mean_series

    for i, yr in enumerate(yr_array):
        col = f"{base_var}_{yr}"
        new_cols[f"{base_var}_anom_{yr}"] = df[col] - mean_series
        if i >= 1:
            prev_col = f"{base_var}_{yr_array[i-1]}"
            new_cols[f"{base_var}_d1_{yr}"] = df[col] - df[prev_col]
        if i >= 3:
            col_3back = f"{base_var}_{yr_array[i-3]}"
            new_cols[f"{base_var}_d3_{yr}"] = (df[col] - df[col_3back]) / 3.0

    new_cols[f"{base_var}_trend"] = df[cols[-1]] - df[cols[0]]

    return pd.concat([df] + [s.rename(k) for k, s in new_cols.items()], axis=1)


def _to_lag_col(col: str, pred_yr: int, feat_yrs: list[int]) -> str:
    """Rename {var}_{YYYY} → {var}_Lag{T-YYYY} for consistent temporal indexing.

    Example: pop_2018 with pred_yr=2019 → pop_Lag1 (1 year before prediction).
             pop_2018 with pred_yr=2022 → pop_Lag4 (4 years before prediction).

    Checks suffix only (longest match first) to avoid partial replacements.
    Columns without a matching year suffix are returned unchanged.
    """
    for yr in sorted(feat_yrs, reverse=True):
        suffix = f"_{yr}"
        if col.endswith(suffix):
            lag = pred_yr - yr
            return col[: -len(suffix)] + f"_Lag{lag}"
    return col


def build_sliding_window_dataset(
    df_features: pd.DataFrame,
    lossyear_raw: pd.Series,
    prediction_years: list[int],
    feature_window: int = 4,
) -> pd.DataFrame:
    """Create one row per (location, prediction_year) using sliding window.

    For each prediction year T, features from years [T-feature_window, T-1]
    are used to predict deforestation in year T.

    Year-indexed columns (e.g. pop_2018) are renamed to lag-indexed columns
    (e.g. pop_Lag1 for pred_yr=2019, pop_Lag4 for pred_yr=2022) so that the
    model always sees the same feature names regardless of which prediction
    year a row belongs to. This prevents train/test distributional shift caused
    by NaN-filled absolute-year columns.

    Locations already deforested before the prediction window are excluded
    (they cannot be deforested again).

    Args:
        df_features: Wide DataFrame with columns like {var}_{year}.
        lossyear_raw: Hansen lossyear (0=none, 1-24=2001-2024) per location.
        prediction_years: Years to predict (e.g. [2019, 2020, 2021, 2022]).
        feature_window: Number of years of history to include.

    Returns:
        DataFrame with columns: pid, prediction_year, target, split,
        static features (original names), and lag-indexed feature columns.
    """
    rows = []

    for pred_yr in prediction_years:
        # Years used as features
        feat_yrs = list(range(pred_yr - feature_window, pred_yr))

        # Target: was this location deforested in pred_yr?
        yr_code = pred_yr - 2000
        target = (lossyear_raw == yr_code).astype(int)

        # Exclude locations already deforested before the feature window
        already_lost_code_max = feat_yrs[0] - 2000 - 1  # before window start
        if already_lost_code_max > 0:
            previously_deforested = lossyear_raw.between(1, already_lost_code_max)
        else:
            previously_deforested = pd.Series(False, index=lossyear_raw.index)

        # Static features: columns with no year substring 2010-2030
        static_cols = [
            c for c in df_features.columns
            if not any(str(y) in c for y in range(2010, 2031))
        ]

        # Year-specific features for this prediction window
        year_cols = []
        for yr in feat_yrs:
            yr_cols = [
                c for c in df_features.columns
                if c.endswith(f"_{yr}") or f"_{yr}_" in c
            ]
            year_cols.extend(yr_cols)
        year_cols = list(dict.fromkeys(year_cols))  # deduplicate, preserve order

        # Select available columns
        available_static = [c for c in static_cols if c in df_features.columns]
        available_year = [c for c in year_cols if c in df_features.columns]

        df_window = df_features[available_static + available_year].copy()

        # Rename year-indexed → lag-indexed for temporal consistency
        rename_map = {
            col: _to_lag_col(col, pred_yr, feat_yrs)
            for col in available_year
        }
        df_window = df_window.rename(columns=rename_map)

        df_window["target"] = target
        df_window["prediction_year"] = pred_yr
        df_window["already_deforested"] = previously_deforested

        # Filter out previously deforested
        df_window = df_window[~df_window["already_deforested"]].drop(
            columns=["already_deforested"]
        )

        # Assign split dynamically: test = last prediction year, val = second-to-last
        test_yr = prediction_years[-1]
        val_yr = prediction_years[-2] if len(prediction_years) > 1 else None
        if pred_yr == test_yr:
            df_window["split"] = "test"
        elif pred_yr == val_yr:
            df_window["split"] = "val"
        else:
            df_window["split"] = "train"

        rows.append(df_window)

    return pd.concat(rows).copy().reset_index().rename(columns={"index": "pid"})


# ─── Convenience: full rebuild from raw parquet ───────────────────────────────


def rebuild_features_dataset(
    df_raw: pd.DataFrame,
    years: list[int],
    prediction_years: list[int],
    feature_window: int = 4,
) -> pd.DataFrame:
    """Rebuild the sliding-window features dataset from a raw parquet DataFrame.

    Auto-detects all annual time-series variables (columns matching {var}_{YYYY}
    where YYYY is in `years`) and computes temporal features (delta, trend,
    anomaly) for each. Static features are passed through unchanged.

    Args:
        df_raw: Raw parquet with columns like {var}_{year}.
        years: Training years with GEE data (e.g. [2016, 2017, ..., 2021]).
        prediction_years: Prediction years for sliding window (e.g. [2019..2022]).
        feature_window: Number of history years per prediction row.

    Returns:
        Sliding-window dataset with lag-indexed feature columns.
    """
    _annual_re = re.compile(r"^(.+)_(\d{4})$")
    base_vars: set[str] = set()
    for col in df_raw.columns:
        m = _annual_re.match(col)
        if m and int(m.group(2)) in years:
            base_vars.add(m.group(1))

    df_eng = df_raw.copy()
    for base in sorted(base_vars):
        df_eng = build_temporal_features(df_eng, base, years)
    # Defragment after many concat operations
    df_eng = df_eng.copy()

    if "lossyear" in df_eng.columns:
        defo_dict: dict[str, pd.Series] = {}
        lossyear = df_eng["lossyear"].fillna(0)
        for yr in years:
            yr_code = yr - 2000
            # Cumulative state: was this pixel already deforested BY year yr?
            # Irreversible process → stays 1 once deforested.
            # d1 of this = new deforestation event in that specific year (meaningful).
            # Replaces per-year indicator (deforested==yr_code) whose d1 is noise
            # (difference of two mutually exclusive binary indicators).
            defo_dict[f"cum_deforested_{yr}"] = lossyear.between(1, yr_code).astype(float)
        defo_dict["cum_loss_before"] = lossyear.between(
            1, years[0] - 2000 - 1
        ).astype(float)
        df_defo = pd.DataFrame(defo_dict, index=df_eng.index)
        # NOTE: Do NOT add global loss_last2yrs / forest_remaining here —
        # they would use cum_deforested_{years[-1]} (= last training year)
        # which equals the val prediction year → direct target leakage.
        # Per-window equivalents are computed in add_window_summaries()
        # from cum_deforested_Lag1 (the most recent lag, always in-window).
        df_eng = pd.concat([df_eng, df_defo], axis=1)
        # Build temporal features on cumulative indicator (d1 = new loss, d3 = acceleration)
        df_eng = build_temporal_features(df_eng, "cum_deforested", years)

    lossyear_raw = df_raw["lossyear"].fillna(0) if "lossyear" in df_raw.columns \
        else pd.Series(0, index=df_raw.index)
    return build_sliding_window_dataset(
        df_eng, lossyear_raw, prediction_years, feature_window
    )
