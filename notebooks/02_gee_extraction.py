import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Deforestation Prediction — Congo Basin
        ## 02 — Google Earth Engine Data Extraction

        Extract multi-dimensional features at sample points using GEE.
        All temporal aggregation is done server-side to minimize EECU usage.
        """
    )
    return (mo,)


@app.cell
def _():
    import ee
    import pandas as pd
    import numpy as np
    from pathlib import Path

    PROJECT_DIR = Path("/Users/guillaume/Documents/Recherche/Deforestation")
    return Path, PROJECT_DIR, ee, np, pd


@app.cell
def _(ee, mo):
    # Initialize GEE — will prompt for authentication on first run
    try:
        ee.Initialize()
        mo.md("**GEE initialized successfully.**")
    except Exception:
        ee.Authenticate()
        ee.Initialize()
        mo.md("**GEE authenticated and initialized.**")
    return


@app.cell
def _(PROJECT_DIR, ee, pd):
    # Load sample points and upload to GEE
    points_df = pd.read_parquet(PROJECT_DIR / "data" / "sample_points.parquet")
    print(f"Loaded {len(points_df):,} sample points")

    # Convert to GEE FeatureCollection (use a subset for testing)
    N_TEST = 1000  # Start small, scale up later
    sample = points_df.sample(N_TEST, random_state=42)

    features = []
    for _, row in sample.iterrows():
        point = ee.Geometry.Point([row["lon"], row["lat"]])
        feat = ee.Feature(point, {"lon": row["lon"], "lat": row["lat"]})
        features.append(feat)

    fc = ee.FeatureCollection(features)
    print(f"Created GEE FeatureCollection with {N_TEST} points")
    return N_TEST, fc, features, points_df, sample


@app.cell
def _(ee, mo):
    mo.md(
        """
        ## Data Sources Available in GEE

        | Dataset | GEE ID | Temporal |
        |---------|--------|----------|
        | Hansen GFC | `UMD/hansen/global_forest_change_2022_v1_10` | Static + lossyear |
        | SRTM | `USGS/SRTMGL1_003` | Static |
        | CHIRPS Daily | `UCSB-CHG/CHIRPS/DAILY` | Daily |
        | ERA5-Land Daily | `ECMWF/ERA5_LAND/DAILY_AGGR` | Daily |
        | VIIRS NTL Monthly | `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG` | Monthly |
        | MODIS Fire | `MODIS/061/MOD14A1` | Daily |
        | WorldPop | `WorldPop/GP/100m/pop` | Annual |
        | Sentinel-2 SR | `COPERNICUS/S2_SR_HARMONIZED` | ~5 days |
        """
    )
    return


@app.cell
def _(ee, fc):
    # === EXTRACTION 1: Static features (SRTM + Hansen treecover) ===

    srtm = ee.Image("USGS/SRTMGL1_003")
    hansen = ee.Image("UMD/hansen/global_forest_change_2022_v1_10")

    static_image = (
        srtm.select("elevation")
        .addBands(ee.Terrain.slope(srtm).rename("slope"))
        .addBands(hansen.select("treecover2000"))
        .addBands(hansen.select("lossyear"))
    )

    # Extract at points
    static_result = static_image.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.first(),
        scale=30,
    )

    # Get info (small test)
    print("Static extraction: bands =", static_image.bandNames().getInfo())
    first = static_result.first().getInfo()
    print("First point properties:", first["properties"])
    return first, hansen, srtm, static_image, static_result


@app.cell
def _(ee, fc):
    # === EXTRACTION 2: Climate temporal profiles (CHIRPS) ===
    # Compute seasonal stats SERVER-SIDE in GEE, then extract

    def chirps_annual_profile(year):
        """Compute annual climate profile from daily CHIRPS data."""
        start = ee.Date.fromYMD(year, 1, 1)
        end = ee.Date.fromYMD(year, 12, 31)

        daily = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(start, end)

        total_precip = daily.sum().rename("precip_total")
        mean_precip = daily.mean().rename("precip_mean")
        max_precip = daily.max().rename("precip_max")

        # Number of dry days (< 1mm)
        dry_days = daily.map(lambda img: img.lt(1).rename("dry")).sum().rename("dry_days")

        # Number of extreme rain days (> 50mm)
        extreme_days = (
            daily.map(lambda img: img.gt(50).rename("extreme")).sum().rename("extreme_rain_days")
        )

        return (
            total_precip.addBands(mean_precip)
            .addBands(max_precip)
            .addBands(dry_days)
            .addBands(extreme_days)
            .set("year", year)
        )

    # Test with one year
    profile_2020 = chirps_annual_profile(2020)
    print("CHIRPS 2020 bands:", profile_2020.bandNames().getInfo())

    # Extract at test points
    chirps_result = profile_2020.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.first(),
        scale=5000,  # CHIRPS is 5km resolution
    )
    first_chirps = chirps_result.first().getInfo()
    print("First point CHIRPS:", first_chirps["properties"])
    return chirps_annual_profile, chirps_result, first_chirps, profile_2020


@app.cell
def _(mo):
    mo.md(
        """
        ## Next Steps

        1. Scale extraction to all 250K points (batch export via `ee.batch.Export`)
        2. Add ERA5-Land (temperature, soil moisture, evapotranspiration)
        3. Add VIIRS nighttime lights (monthly → annual profiles)
        4. Add MODIS fire (fire frequency, seasonality)
        5. Add WorldPop (population density → temporal profiles)
        6. Extract at multiple spatial buffers (50m, 150m, 500m, 1km, 3km)
        7. Compute temporal profiles (Δ1yr, Δ3yr, Δ10yr, anomaly) for each variable
        """
    )
    return


if __name__ == "__main__":
    app.run()
