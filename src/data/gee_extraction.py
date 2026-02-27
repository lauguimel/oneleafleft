"""GEE extraction utilities for the deforestation prediction pipeline.

Design principles:
- All temporal aggregation happens server-side in GEE (minimize EECU)
- Points are batched in chunks of BATCH_SIZE (GEE limit ~5000 features)
- Each function returns a pd.DataFrame indexed by point ID
- NaN = no data at that location for that source/year
"""

from __future__ import annotations

import logging
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
) -> pd.DataFrame:
    """Extract an ee.Image at all points, batching automatically.

    Args:
        image: Single ee.Image with one or more bands already named.
        points: DataFrame with columns [lon, lat], indexed by pid.
        scale: Spatial resolution in metres.
        batch_size: Max features per GEE call.
        verbose: Print progress.

    Returns:
        DataFrame indexed by pid with one column per image band.
    """
    n = len(points)
    n_batches = (n + batch_size - 1) // batch_size
    results = []

    for i in range(n_batches):
        batch = points.iloc[i * batch_size : (i + 1) * batch_size]
        fc = _fc_from_df(batch, pid_col=pid_col)
        df_batch = _extract_batch(image, fc, scale, pid_col=pid_col)
        results.append(df_batch)
        if verbose and n_batches > 1:
            log.info(f"  Batch {i+1}/{n_batches} done ({len(batch)} pts)")

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


# ─── WorldPop annual (COD, 2000-2020) ────────────────────────────────────────


def extract_worldpop(
    points: pd.DataFrame,
    years: list[int],
    country: str = "COD",
    **kw,
) -> pd.DataFrame:
    """WorldPop population density (persons/pixel) for requested years.

    Data available for COD: 2000-2020 (band: 'population').
    Years beyond 2020 are clamped to 2020.
    """
    worldpop = ee.ImageCollection("WorldPop/GP/100m/pop")
    dfs = {}
    for yr in years:
        yr_eff = min(yr, 2020)
        img = (
            worldpop
            .filter(ee.Filter.eq("country", country))
            .filter(ee.Filter.eq("year", yr_eff))
            .first()
            .select("population")
            .rename(f"pop_{yr}")
        )
        df = extract_image(img, points, scale=100, **kw)
        # GEE may rename band: find the right column
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


# ─── Multi-buffer spatial aggregation ────────────────────────────────────────


def extract_hansen_buffers(
    points: pd.DataFrame,
    years: list[int],
    buffers_m: list[int] | None = None,
    **kw,
) -> pd.DataFrame:
    """Deforestation rate within spatial buffers around each point.

    For each buffer radius and each year, compute the fraction of
    pixels deforested in that year within the buffer zone.

    Args:
        points: DataFrame with lon/lat, indexed by pid.
        years: List of years to compute deforestation rates for.
        buffers_m: Buffer radii in metres (default: [150, 500, 1500, 5000]).

    Returns:
        DataFrame with columns like defo_rate_150m_2019, defo_rate_5000m_2020.
    """
    if buffers_m is None:
        buffers_m = [150, 500, 1500, 5000]

    hansen = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")

    all_dfs = []
    for yr in years:
        yr_code = yr - 2000  # Hansen lossyear encoding
        loss_yr = hansen.select("lossyear").eq(yr_code).rename("loss")

        for radius in buffers_m:
            # Mean over buffer = fraction of pixels deforested
            loss_buffered = loss_yr.reduceNeighborhood(
                reducer=ee.Reducer.mean(),
                kernel=ee.Kernel.circle(radius, "meters"),
            ).rename(f"defo_rate_{radius}m_{yr}")

            df = extract_image(loss_buffered, points, scale=30, **kw)
            all_dfs.append(df)

    return pd.concat(all_dfs, axis=1)


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
    df = df.copy()
    cols = [f"{base_var}_{yr}" for yr in years if f"{base_var}_{yr}" in df.columns]
    if len(cols) < 2:
        return df

    yr_array = [yr for yr in years if f"{base_var}_{yr}" in df.columns]

    # Mean across years (for anomaly computation)
    df[f"{base_var}_mean"] = df[cols].mean(axis=1)

    for i, yr in enumerate(yr_array):
        col = f"{base_var}_{yr}"
        # Anomaly vs mean
        df[f"{base_var}_anom_{yr}"] = df[col] - df[f"{base_var}_mean"]
        # Δ1yr
        if i >= 1:
            prev_col = f"{base_var}_{yr_array[i-1]}"
            df[f"{base_var}_d1_{yr}"] = df[col] - df[prev_col]
        # Δ3yr (slope over last 3 years)
        if i >= 3:
            col_3back = f"{base_var}_{yr_array[i-3]}"
            df[f"{base_var}_d3_{yr}"] = (df[col] - df[col_3back]) / 3.0

    # Long-term trend (first to last)
    df[f"{base_var}_trend"] = df[cols[-1]] - df[cols[0]]

    return df


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

        # Static features: columns with no year substring 2010-2024
        static_cols = [
            c for c in df_features.columns
            if not any(str(y) in c for y in range(2010, 2025))
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

        # Assign split: train ≤2020, val=2021, test=2022
        if pred_yr <= 2020:
            df_window["split"] = "train"
        elif pred_yr == 2021:
            df_window["split"] = "val"
        else:
            df_window["split"] = "test"

        rows.append(df_window)

    return pd.concat(rows).reset_index().rename(columns={"index": "pid"})
