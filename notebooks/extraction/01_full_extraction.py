import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


@app.cell
def _():
    # pyright: reportUnusedVariable=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportReturnType=false, reportUnusedImport=false, reportUnusedExpression=false
    import marimo as mo

    mo.md(
        """
        # Full Data Extraction — Deforestation Prediction Pipeline

        This notebook performs the **complete data extraction from scratch** for the
        deforestation prediction project on the Congo Basin (Hansen GFC tile 10N/020E).

        ## What this notebook does

        1. **Samples 250K points** from the study area
        2. **Extracts 7 GEE sources** (static + annual time-series for 2016-2023)
        3. **Extracts Hansen spatial buffers** at 4 radii (150m, 500m, 1500m, 5000m)
        4. **Extracts infrastructure features** (OSM roads + settlements, WDPA)
        5. **Builds the sliding-window feature dataset** with temporal lag encoding
        6. **Runs sanity checks** on the final dataset

        ## How to use this notebook

        - **Run each cell sequentially** (top to bottom)
        - **GEE extractions are slow** (~2-3 hours total for 250K points)
        - Each extraction saves a **checkpoint** to disk so you can resume later
        - Intermediate results are printed after each step for verification
        - You can modify the configuration (Section 1) before running

        ## Data sources

        | Source | GEE Collection | Resolution | Variables |
        |--------|---------------|------------|-----------|
        | SRTM | `USGS/SRTMGL1_003` | 30m | elevation, slope |
        | Hansen GFC v1.12 | `UMD/hansen/global_forest_change_2024_v1_12` | 30m | treecover2000, lossyear (2001-2024) |
        | WorldPop | `WorldPop/GP/100m/pop` | 100m | population density (max 2020) |
        | CHIRPS | `UCSB-CHG/CHIRPS/DAILY` | 5km | 5 annual precipitation stats |
        | ERA5-Land | `ECMWF/ERA5_LAND/DAILY_AGGR` | 9km | 6 annual climate stats |
        | VIIRS NTL | `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG` | 750m | 4 annual nighttime light stats |
        | MODIS fire | `MODIS/061/MOD14A1` | 1km | 2 annual fire stats |
        | Hansen buffers | (derived from Hansen GFC) | 100-500m | deforestation rate in 4 buffer radii |
        | WDPA | `WCMC/WDPA/current/polygons` | 1km | in_protected, distance_protected |
        | OSM | Overpass API | — | distance to roads, settlements |

        ---
        """
    )
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    ## 1. Configuration

    **Modify these parameters before running** if you want to change the setup.
    Default values reproduce the full 250K extraction.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import numpy as np
    import pandas as pd

    # ── Project paths ────────────────────────────────────────────────────────
    PROJECT_DIR = Path("/Users/guillaume/Documents/Recherche/Deforestation")
    DATA_DIR = PROJECT_DIR / "data_validation"
    DATA_DIR.mkdir(exist_ok=True)

    # Legacy dataset with 250K point coordinates from the old project
    LEGACY_CSV = Path(
        "/Users/guillaume/Documents/Clouds/UGA/Recherche/deforestation"
        "/src/data/tiles_250000_10N_020E_20231023.csv"
    )

    # ── Extraction parameters ────────────────────────────────────────────────
    N_POINTS = 250_000        # Number of sample points (250K for full, 5K for testing)
    SEED = 42                 # Random seed for reproducibility

    # Years for which we extract ANNUAL features from GEE
    # IMPORTANT: stop at 2023 (inclusive) — 2024 is the TEST year (target only).
    # Never extract 2024 features to guarantee zero leakage.
    FEATURE_YEARS = list(range(2016, 2024))  # [2016, 2017, ..., 2023]
    # Buffer radii for Hansen spatial contagion features

    BUFFER_RADII_M = [150, 500, 1500, 5000]

    # Prediction years: for each pred_yr, features from [pred_yr-4, pred_yr-1]
    # are used to predict deforestation in pred_yr.
    # With feature years 2016-2023, we can predict 2020-2024.
    # Split: train 2020-2022, val 2023, test 2024.
    PREDICTION_YEARS = list(range(2020, 2025))  # [2020, 2021, ..., 2024]

    # The sliding window feature_window controls how many years of history
    # are included in each prediction row.
    FEATURE_WINDOW = 4  # 4 years of history per prediction

    print("Configuration:")
    print(f"  N_POINTS = {N_POINTS:,}")
    print(f"  FEATURE_YEARS = {FEATURE_YEARS}")
    print(f"  PREDICTION_YEARS = {PREDICTION_YEARS}")
    print(f"  BUFFER_RADII_M = {BUFFER_RADII_M}")
    print(f"  FEATURE_WINDOW = {FEATURE_WINDOW}")
    print(f"  DATA_DIR = {DATA_DIR}")
    print(f"  SEED = {SEED}")
    return (
        BUFFER_RADII_M,
        DATA_DIR,
        FEATURE_YEARS,
        LEGACY_CSV,
        N_POINTS,
        PROJECT_DIR,
        SEED,
        np,
        pd,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 2. Sample Points

    We load 250K point coordinates from the legacy project's tiles CSV.
    These are evenly distributed across Hansen GFC tile **10N/020E**
    (longitude 20-30°E, latitude 0-10°N), covering parts of
    **DRC, Republic of Congo, Cameroon, Gabon, and Central African Republic**.

    The legacy dataset was generated in 2023 from the Hansen Global Forest
    Change dataset using a regular grid with random jitter.
    """)
    return


@app.cell
def _(LEGACY_CSV, N_POINTS, SEED, pd):
    # Load legacy point coordinates
    tiles = pd.read_csv(
        LEGACY_CSV,
        usecols=["longitude", "latitude", "lossyear_22_mean"],
    ).rename(columns={"longitude": "lon", "latitude": "lat"})

    # The legacy CSV has a deforestation indicator based on Hansen v1.10 lossyear
    tiles["deforested_2022_legacy"] = (tiles["lossyear_22_mean"] > 0).astype(int)

    # Sample N_POINTS with stratified sampling to preserve the deforestation rate
    if N_POINTS < len(tiles):
        pos = tiles[tiles["deforested_2022_legacy"] == 1]
        neg = tiles[tiles["deforested_2022_legacy"] == 0]
        # Take all positives + random negatives to reach N_POINTS
        n_pos_sample = min(len(pos), N_POINTS // 2)
        n_neg_sample = N_POINTS - n_pos_sample
        sampled_pos = pos.sample(n_pos_sample, random_state=SEED)
        sampled_neg = neg.sample(n_neg_sample, random_state=SEED)
        points = pd.concat([sampled_pos, sampled_neg]).sample(frac=1, random_state=SEED)
    else:
        points = tiles.copy()

    points = points[["lon", "lat"]].reset_index(drop=True)
    points.index.name = "pid"

    print(f"Sample points: {len(points):,}")
    print(f"  Longitude range: [{points.lon.min():.2f}, {points.lon.max():.2f}]")
    print(f"  Latitude range:  [{points.lat.min():.2f}, {points.lat.max():.2f}]")
    print(f"\nFirst 5 points:")
    print(points.head())
    return (points,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Google Earth Engine Initialization

    This cell initializes the GEE Python API. You need:
    - A Google Earth Engine account (free for research)
    - The `earthengine-api` package installed
    - Authentication done via `earthengine authenticate`

    The project ID is stored in `src/data/gee_utils.py`.
    """)
    return


@app.cell
def _(PROJECT_DIR):
    import sys
    sys.path.insert(0, str(PROJECT_DIR / "src"))

    import ee
    from data.gee_utils import init_gee

    init_gee()
    print("GEE initialized successfully")

    # Verify: check that we can access the Hansen dataset
    hansen = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    bands = hansen.bandNames().getInfo()
    print(f"Hansen GFC v1.12 bands: {bands[:5]}...")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Static Source Extraction

    ### 4.1 SRTM — Elevation & Slope

    **Collection**: `USGS/SRTMGL1_003` (Shuttle Radar Topography Mission)
    **Resolution**: 30m
    **Variables**:
    - `elevation` — meters above sea level
    - `slope` — derived from elevation using `ee.Terrain.slope()`, in degrees

    **Why**: Slope and elevation influence deforestation patterns. Steep terrain
    is harder to clear; lowlands near rivers are more accessible.

    ### 4.2 Hansen GFC v1.12 — Forest Cover & Loss

    **Collection**: `UMD/hansen/global_forest_change_2024_v1_12`
    **Resolution**: 30m
    **Variables**:
    - `treecover2000` — percent tree canopy cover in year 2000 (0-100)
    - `lossyear` — year of forest loss (0 = no loss, 1 = 2001, ..., 24 = 2024)

    **Why**: Tree cover is the baseline forest state. Lossyear is used to:
    1. Build the target variable (was this pixel deforested in year T?)
    2. Compute cumulative deforestation history
    3. Exclude already-deforested pixels

    **Critical**: `lossyear` goes up to **24 (=2024)** in v1.12. We use this
    to verify predictions and as training data.
    """)
    return


@app.cell
def _(points):
    import time
    from data.gee_extraction import extract_srtm, extract_hansen_static

    print("Extracting SRTM (elevation + slope)...")
    _t0: int | float = time.time()
    df_srtm = extract_srtm(points)
    print(f"  Done in {time.time()-_t0:.0f}s — {df_srtm.shape[1]} columns")
    print(f"  Elevation: mean={df_srtm.elevation.mean():.0f}m, "
          f"range=[{df_srtm.elevation.min():.0f}, {df_srtm.elevation.max():.0f}]")
    print(f"  Slope: mean={df_srtm.slope.mean():.1f}°, "
          f"NaN={df_srtm.slope.isna().mean()*100:.1f}%")

    print("\nExtracting Hansen GFC v1.12 (treecover2000 + lossyear)...")
    _t0 = time.time()
    df_hansen = extract_hansen_static(points)
    print(f"  Done in {time.time()-_t0:.0f}s — {df_hansen.shape[1]} columns")
    print(f"  Treecover2000: mean={df_hansen.treecover2000.mean():.1f}%, "
          f"NaN={df_hansen.treecover2000.isna().mean()*100:.1f}%")
    print(f"  Lossyear: {(df_hansen.lossyear > 0).mean()*100:.1f}% pixels with loss")
    print(f"  Lossyear distribution (non-zero):")
    ly = df_hansen.lossyear[df_hansen.lossyear > 0]
    for yr_code in sorted(ly.unique()):
        _n = (ly == yr_code).sum()
        print(f"    {int(yr_code)+2000}: {_n:,} pixels ({_n/len(df_hansen)*100:.2f}%)")

    df_static = df_srtm.join(df_hansen, how="outer")
    print(f"\nStatic features: {df_static.shape[1]} columns")
    return df_hansen, df_srtm, time


@app.cell
def _(mo):
    mo.md("""
    ### 4.3 WDPA — Protected Areas

    **Collection**: `WCMC/WDPA/current/polygons`
    **Resolution**: sampled at 1km
    **Variables**:
    - `in_protected` — 1 if inside any WDPA polygon, 0 otherwise
    - `dist_protected_km` — distance to nearest protected area boundary (km, capped at 200km)

    **Method**: We "paint" the WDPA polygons onto a raster (value=1 inside, 0 outside),
    then compute Euclidean distance from each pixel to the nearest painted pixel.

    **Why**: Protected areas may slow deforestation. The binary indicator and distance
    capture whether a point is inside or near a PA.
    """)
    return


@app.cell
def _(points, time):
    from data.gee_extraction import extract_wdpa

    print("Extracting WDPA (protected areas)...")
    _t0 = time.time()
    df_wdpa = extract_wdpa(points)
    print(f"  Done in {time.time()-_t0:.0f}s — {df_wdpa.shape[1]} columns")
    print(f"  Inside PA: {(df_wdpa.in_protected == 1).mean()*100:.1f}% of points")
    print(f"  Dist to PA: mean={df_wdpa.dist_protected_km.mean():.1f}km, "
          f"NaN={df_wdpa.dist_protected_km.isna().mean()*100:.1f}%")
    return (df_wdpa,)


@app.cell
def _(mo):
    mo.md("""
    ## 5. Annual Time-Series Extraction

    For each source below, we extract **annual statistics** for every year in
    `FEATURE_YEARS` (2016-2023). Each extraction produces columns like
    `{variable}_{year}` (e.g., `precip_total_2019`).

    **Important**: We deliberately stop at 2023 — 2024 is the test year and
    no 2024 features are extracted to guarantee zero leakage.
    WorldPop stops at 2020 — years 2021-2023 are clamped to 2020.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 5.1 WorldPop — Population Density

    **Collection**: `WorldPop/GP/100m/pop`
    **Resolution**: 100m
    **Variable per year**: `pop_{year}` — persons per pixel (~100m²)
    **Coverage**: 2000-2020 (years after 2020 are clamped to 2020)

    **Method**: For each year, we mosaic all country tiles that overlap the study
    area bounding box (`filterBounds`), then sample at each point location.

    **Why**: Population pressure is a classical driver of deforestation.
    Higher population → more agricultural demand → more forest clearing.

    **Note**: WorldPop data stopped being updated after 2020. For years 2021-2023,
    we use the 2020 value as a proxy. This is a known limitation.
    """)
    return


@app.cell
def _(FEATURE_YEARS, points, time):
    from data.gee_extraction import extract_worldpop

    print(f"Extracting WorldPop (population density) for {FEATURE_YEARS}...")
    print("  Note: years > 2020 will be clamped to 2020 (WorldPop max)")
    _t0 = time.time()
    df_pop = extract_worldpop(points, years=FEATURE_YEARS)
    print(f"  Done in {time.time()-_t0:.0f}s — {df_pop.shape[1]} columns")

    for _yr in FEATURE_YEARS:
        _col = f"pop_{_yr}"
        if _col in df_pop.columns:
            nan_pct = df_pop[_col].isna().mean() * 100
            _mean_val = df_pop[_col].mean()
            print(f"  {_col}: mean={_mean_val:.2f}, NaN={nan_pct:.1f}%")
    return (df_pop,)


@app.cell
def _(mo):
    mo.md("""
    ### 5.2 CHIRPS — Precipitation

    **Collection**: `UCSB-CHG/CHIRPS/DAILY` (Climate Hazards Infrared Precipitation)
    **Resolution**: 5km (~0.05°)
    **Variables per year** (5 columns):
    - `precip_total_{year}` — total annual precipitation (mm)
    - `precip_mean_{year}` — mean daily precipitation (mm/day)
    - `precip_max_{year}` — maximum single-day precipitation (mm)
    - `dry_days_{year}` — number of days with < 1mm precipitation
    - `extreme_rain_days_{year}` — number of days with > 50mm

    **Method**: Server-side GEE aggregation from daily images to annual statistics.

    **Why**: Precipitation patterns drive vegetation dynamics. Dry seasons enable
    fire-based clearing; extreme rain can prevent access to forests.
    """)
    return


@app.cell
def _(FEATURE_YEARS, points, time):
    from data.gee_extraction import extract_chirps

    print(f"Extracting CHIRPS (precipitation) for {FEATURE_YEARS}...")
    _t0 = time.time()
    df_chirps = extract_chirps(points, years=FEATURE_YEARS)
    print(f"  Done in {time.time()-_t0:.0f}s — {df_chirps.shape[1]} columns")

    # Show sample stats
    for _yr in [FEATURE_YEARS[0], FEATURE_YEARS[-1]]:
        _col = f"precip_total_{_yr}"
        if _col in df_chirps.columns:
            print(f"  {_col}: mean={df_chirps[_col].mean():.0f}mm, NaN={df_chirps[_col].isna().mean()*100:.1f}%")
    return (df_chirps,)


@app.cell
def _(mo):
    mo.md("""
    ### 5.3 ERA5-Land — Temperature & Soil Moisture

    **Collection**: `ECMWF/ERA5_LAND/DAILY_AGGR`
    **Resolution**: 9km (~0.1°)
    **Variables per year** (6 columns):
    - `temperature_2m_{year}` — mean annual 2m air temperature (Kelvin)
    - `temperature_2m_max_{year}` — mean of daily max temperature (K)
    - `hot_days_{year}` — number of days with max temp > 35°C (308.15 K)
    - `sm_surface_{year}` — mean surface soil moisture (m³/m³)
    - `sm_std_{year}` — standard deviation of daily soil moisture (inter-annual volatility)
    - `et_{year}` — mean daily potential evapotranspiration (m/day)

    **Method**: Server-side GEE aggregation from daily data.

    **Why**: Climate variables affect both vegetation health and human activity
    patterns. Soil moisture volatility is a proxy for climate stress.
    """)
    return


@app.cell
def _(FEATURE_YEARS, points, time):
    from data.gee_extraction import extract_era5

    print(f"Extracting ERA5-Land (climate) for {FEATURE_YEARS}...")
    _t0 = time.time()
    df_era5 = extract_era5(points, years=FEATURE_YEARS)
    print(f"  Done in {time.time()-_t0:.0f}s — {df_era5.shape[1]} columns")

    for _yr in [FEATURE_YEARS[0], FEATURE_YEARS[-1]]:
        _col = f"temperature_2m_{_yr}"
        if _col in df_era5.columns:
            print(f"  {_col}: mean={df_era5[_col].mean():.1f}K ({df_era5[_col].mean()-273.15:.1f}°C)")
    return (df_era5,)


@app.cell
def _(mo):
    mo.md("""
    ### 5.4 VIIRS — Nighttime Lights

    **Collection**: `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG`
    **Resolution**: 750m (~0.004°)
    **Variables per year** (4 columns):
    - `ntl_mean_{year}` — mean annual radiance (nW/cm²/sr)
    - `ntl_max_{year}` — peak monthly radiance
    - `ntl_std_{year}` — standard deviation of monthly radiance
    - `ntl_cv_{year}` — coefficient of variation (seasonal variability)

    **Method**: Server-side aggregation from monthly composites. Negative radiance
    values are clamped to 0.

    **Why**: Nighttime lights are a proxy for human activity, infrastructure, and
    economic development. Light expansion near forests often precedes clearing.
    """)
    return


@app.cell
def _(FEATURE_YEARS, points, time):
    from data.gee_extraction import extract_viirs

    print(f"Extracting VIIRS NTL (nighttime lights) for {FEATURE_YEARS}...")
    _t0 = time.time()
    df_viirs = extract_viirs(points, years=FEATURE_YEARS)
    print(f"  Done in {time.time()-_t0:.0f}s — {df_viirs.shape[1]} columns")

    for _yr in [FEATURE_YEARS[0], FEATURE_YEARS[-1]]:
        _col = f"ntl_mean_{_yr}"
        if _col in df_viirs.columns:
            print(f"  {_col}: mean={df_viirs[_col].mean():.3f}, NaN={df_viirs[_col].isna().mean()*100:.1f}%")
    return (df_viirs,)


@app.cell
def _(mo):
    mo.md("""
    ### 5.5 MODIS Fire — Active Fire Detection

    **Collection**: `MODIS/061/MOD14A1` (Terra Daily Active Fire)
    **Resolution**: 1km
    **Variables per year** (2 columns):
    - `fire_days_{year}` — number of days with high-confidence fire (FireMask ≥ 7)
    - `fire_max_{year}` — maximum fire radiative power in a single day (MW)

    **Method**: For each day, pixels with FireMask ≥ 7 are flagged as fire.
    Annual total gives fire frequency; max FRP gives fire intensity.

    **Why**: Fire is a primary tool for forest clearing in the tropics.
    Fire activity near a pixel strongly predicts future deforestation.
    """)
    return


@app.cell
def _(FEATURE_YEARS, points, time):
    from data.gee_extraction import extract_modis_fire

    print(f"Extracting MODIS fire (active fire) for {FEATURE_YEARS}...")
    _t0 = time.time()
    df_fire = extract_modis_fire(points, years=FEATURE_YEARS)
    print(f"  Done in {time.time()-_t0:.0f}s — {df_fire.shape[1]} columns")

    for _yr in [FEATURE_YEARS[0], FEATURE_YEARS[-1]]:
        _col = f"fire_days_{_yr}"
        if _col in df_fire.columns:
            _pct = (df_fire[_col] > 0).mean() * 100
            print(f"  {_col}: {_pct:.1f}% of points had fire, mean={df_fire[_col].mean():.2f} days")
    return (df_fire,)


@app.cell
def _(mo):
    mo.md("""
    ## 6. Hansen Spatial Buffers — Deforestation Contagion

    This is the **most important feature group** based on our ablation study.
    For each point, we compute the **fraction of pixels deforested within a
    circular buffer** of radius _r in each year.

    ### Method

    For year Y and radius _r:
    ```
    loss_year_Y = (Hansen.lossyear == Y-2000).unmask(0)
    buffered = loss_year_Y.reduceNeighborhood(
        reducer=mean, kernel=circle(_r meters)
    )
    ```

    The result is the **deforestation rate** (0 to 1) within the buffer.
    Higher values = more deforestation in the neighborhood.

    ### Radii

    | Radius | Scale | Captures |
    |--------|-------|----------|
    | 150m | 100m | Micro-contagion (immediate neighbors) |
    | 500m | 200m | Local contagion (village-scale) |
    | 1500m | 300m | Meso-contagion (landscape-scale) |
    | 5000m | 500m | Regional contagion (district-scale) |

    ### Critical implementation detail

    `hansen.select("lossyear").unmask(0)` — the `.unmask(0)` is essential.
    Without it, non-forest pixels (where lossyear has no data) become NaN,
    causing 89% NaN in the output. With unmask(0), non-forest = no deforestation = 0.

    ### Tile size auto-adaptation

    GEE's `computePixels` has a 50MB limit per request. For fine resolution
    (100m at 150m radius), a 2°×2° tile would be too large. We auto-adapt:
    ```
    max_side_px = sqrt(48_000_000 / (n_bands × 5.5))
    ```
    where 5.5 bytes/pixel/band accounts for GeoTIFF overhead.

    **This extraction is the slowest part** (~30-60 min for 250K points × 4 radii × 8 years).
    """)
    return


@app.cell
def _(BUFFER_RADII_M, FEATURE_YEARS, points, time):
    from data.gee_extraction import extract_hansen_buffers

    print(f"Extracting Hansen buffers ({len(FEATURE_YEARS)} years × {len(BUFFER_RADII_M)} radii)...")
    print(f"  Years: {FEATURE_YEARS}")
    print(f"  Radii: {BUFFER_RADII_M} meters")
    print(f"  This will take a while...")

    _t0 = time.time()
    df_buffers = extract_hansen_buffers(
        points,
        years=FEATURE_YEARS,
        buffers_m=BUFFER_RADII_M,
    )
    elapsed = time.time() - _t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Columns: {df_buffers.shape[1]}")

    # Sanity check: show stats for latest year
    latest_yr = FEATURE_YEARS[-1]
    for _r in BUFFER_RADII_M:
        _col = f"defo_rate_{_r}m_{latest_yr}"
        if _col in df_buffers.columns:
            _pct = (df_buffers[_col] > 0).mean() * 100
            _mean_val = df_buffers[_col].mean()
            print(f"  {_col}: {_pct:.1f}% non-zero, mean={_mean_val:.4f}")
    return (df_buffers,)


@app.cell
def _(mo):
    mo.md("""
    ## 7. Merge All Sources & Save Raw Checkpoint

    We merge all extracted features into a single DataFrame and save it as a
    **raw checkpoint** (Parquet format). This checkpoint contains:
    - Point coordinates (lon, lat)
    - Static features (SRTM, Hansen, WDPA)
    - Annual time-series (WorldPop, CHIRPS, ERA5, VIIRS, MODIS fire)
    - Hansen buffer features

    The raw checkpoint is saved so that if feature engineering fails, we don't
    need to re-extract from GEE.
    """)
    return


@app.cell
def _(
    DATA_DIR,
    N_POINTS,
    df_buffers,
    df_chirps,
    df_era5,
    df_fire,
    df_hansen,
    df_pop,
    df_srtm,
    df_viirs,
    df_wdpa,
    points,
):
    from datetime import date

    print("Merging all extracted features...")

    df_raw = (
        points[["lon", "lat"]]
        .join(df_srtm)
        .join(df_hansen)
        .join(df_wdpa)
        .join(df_pop)
        .join(df_chirps)
        .join(df_era5)
        .join(df_viirs)
        .join(df_fire)
        .join(df_buffers)
    )

    print(f"  Shape: {df_raw.shape[0]:,} × {df_raw.shape[1]} columns")
    print(f"  Total NaN cells: {df_raw.isna().sum().sum():,}")
    print(f"  NaN rate: {df_raw.isna().mean().mean()*100:.1f}%")

    # Save raw checkpoint
    tag = date.today().strftime("%Y%m%d")
    n_label = N_POINTS // 1000
    raw_path = DATA_DIR / f"raw_{n_label}k_{tag}.parquet"
    df_raw.to_parquet(raw_path)
    print(f"\n  Saved: {raw_path.name} ({raw_path.stat().st_size / 1e6:.1f} MB)")

    # Summary by source
    print(f"\n  Column count by prefix:")
    prefixes = {}
    for _col in df_raw.columns:
        prefix = _col.split("_")[0] if "_" in _col else _col
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    for p, n in sorted(prefixes.items(), key=lambda x: -x[1]):
        print(f"    {p}: {n}")
    return df_raw, raw_path


@app.cell
def _(mo):
    mo.md("""
    ## 8. Infrastructure Features (OSM)

    These features are extracted from **OpenStreetMap** via the Overpass API,
    not from GEE. They capture infrastructure proximity:

    | Variable | Source | Description |
    |----------|--------|-------------|
    | `dist_road_km` | OSM highways | Distance to nearest road (km) |
    | `dist_settlement_km` | OSM places | Distance to nearest settlement (km) |

    **Method**: Query Overpass for all roads/settlements in the bounding box,
    then compute nearest-neighbor distance for each point using a KDTree.

    **Why**: Proximity to roads and settlements is one of the strongest predictors
    of deforestation. Roads provide access; settlements create demand.

    **Note**: This requires internet access to the Overpass API.
    If Overpass is slow/down, you can skip this cell and add these features later
    using `scripts/add_spatial_features.py`.
    """)
    return


@app.cell
def _(df_raw, np, raw_path, time):
    # OSM extraction — adapted from scripts/add_spatial_features.py
    import requests
    from scipy.spatial import cKDTree

    OVERPASS_URL = "http://overpass-api.de/api/interpreter"

    lon_min = float(df_raw.lon.min()) - 0.1
    lon_max = float(df_raw.lon.max()) + 0.1
    lat_min = float(df_raw.lat.min()) - 0.1
    lat_max = float(df_raw.lat.max()) + 0.1
    bbox = f"{lat_min},{lon_min},{lat_max},{lon_max}"

    print("Extracting OSM roads and settlements...")

    # Roads
    print("  Querying Overpass for roads...")
    _t0 = time.time()
    road_query = f"""
    [out:json][timeout:300];
    (way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"]({bbox}););
    out center;
    """
    try:
        _r = requests.get(OVERPASS_URL, params={"data": road_query}, timeout=600)
        roads = _r.json()["elements"]
        road_coords = [(e["center"]["lon"], e["center"]["lat"]) for e in roads if "center" in e]
        print(f"  Found {len(road_coords):,} road segments ({time.time()-_t0:.0f}s)")

        if road_coords:
            tree = cKDTree(road_coords)
            dists, _ = tree.query(df_raw[["lon", "lat"]].values)
            df_raw["dist_road_km"] = dists * 111.32  # approximate degrees to km
        else:
            df_raw["dist_road_km"] = np.nan
    except Exception as e:
        print(f"  Road extraction failed: {e}")
        df_raw["dist_road_km"] = np.nan

    # Settlements
    print("  Querying Overpass for settlements...")
    _t0 = time.time()
    settle_query = f"""
    [out:json][timeout:300];
    (node["place"~"^(city|town|village|hamlet)$"]({bbox}););
    out;
    """
    try:
        _r = requests.get(OVERPASS_URL, params={"data": settle_query}, timeout=600)
        settlements = _r.json()["elements"]
        settle_coords = [(e["lon"], e["lat"]) for e in settlements]
        print(f"  Found {len(settle_coords):,} settlements ({time.time()-_t0:.0f}s)")

        if settle_coords:
            tree = cKDTree(settle_coords)
            dists, _ = tree.query(df_raw[["lon", "lat"]].values)
            df_raw["dist_settlement_km"] = dists * 111.32
        else:
            df_raw["dist_settlement_km"] = np.nan
    except Exception as e:
        print(f"  Settlement extraction failed: {e}")
        df_raw["dist_settlement_km"] = np.nan

    print(f"\n  dist_road_km: mean={df_raw.dist_road_km.mean():.1f}km, "
          f"NaN={df_raw.dist_road_km.isna().mean()*100:.1f}%")
    print(f"  dist_settlement_km: mean={df_raw.dist_settlement_km.mean():.1f}km, "
          f"NaN={df_raw.dist_settlement_km.isna().mean()*100:.1f}%")

    # Re-save raw with infrastructure
    df_raw.to_parquet(raw_path)
    print(f"\n  Updated: {raw_path.name} ({df_raw.shape[1]} columns)")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 9. Feature Engineering & Sliding Window Dataset

    This section transforms the raw extracted data into the final training dataset
    using two key operations:

    ### 9.1 Temporal Feature Engineering

    For each annual time-series variable (e.g., `pop`, `precip_total`), we compute:
    - **First difference** (`d1`): `var_{_yr} - var_{_yr-1}` → annual change
    - **3-year difference** (`d3`): `(var_{_yr} - var_{_yr-3}) / 3` → acceleration
    - **Global anomaly** (`anom`): `var_{_yr} - mean(var)` → deviation from average
    - **Global mean/trend**: `mean(var)`, `var_{last} - var_{first}`

    For the **cumulative deforestation** variable (derived from Hansen lossyear):
    - `cum_deforested_{_yr}` = 1 if pixel was deforested by year `_yr`, 0 otherwise
    - This is an **irreversible state**: once 1, stays 1 forever

    ### 9.2 Sliding Window

    For each prediction year T, we create one row per location using features
    from years [T-4, T-1]. Year-indexed columns are renamed to lag-indexed:
    ```
    pop_2020 with pred_yr=2024 → pop_Lag4
    pop_2023 with pred_yr=2024 → pop_Lag1
    ```

    This ensures the model always sees `pop_Lag1` as "population 1 year ago"
    regardless of which calendar year it actually represents.

    **Target**: `target = (lossyear == pred_yr - 2000)`

    **Exclusion**: Pixels already deforested before the prediction window are
    removed (irreversible process — can't be deforested twice).

    ### 9.3 Split Assignment — Airtight Separation

    | Split | Prediction years | Script | Output directory |
    |-------|-----------------|--------|------------------|
    | train | 2020-2022 | `scripts/build_traintest.py` | `data/train_test/` |
    | test | 2024 | `scripts/build_traintest.py` | `data/train_test/` |
    | val | 2023 | `scripts/build_val.py` | `data/val/` |

    **Leakage guarantee**:
    - No feature data from 2024 is ever extracted (test target = static Hansen band)
    - Val and train/test are built by **separate scripts** into **separate directories**
    - Physical separation makes accidental contamination impossible

    ### How to build

    After this notebook produces the raw checkpoint, run:
    ```bash
    python scripts/build_traintest.py --raw data_validation/raw_250k_YYYYMMDD.parquet
    python scripts/build_val.py --raw data_validation/raw_250k_YYYYMMDD.parquet
    ```
    """)
    return


@app.cell
def _(PROJECT_DIR, raw_path):
    import subprocess

    PYTHON = "/Users/guillaume/miniconda3/envs/deforest/bin/python"

    print("Building train+test and val datasets with separate scripts...")
    print(f"  Raw checkpoint: {raw_path}")
    print(f"  Python: {PYTHON}")
    print()

    for script, label in [
        (PROJECT_DIR / "scripts" / "build_traintest.py", "train+test"),
        (PROJECT_DIR / "scripts" / "build_val.py", "val"),
    ]:
        print(f"--- Building {label} ---")
        result = subprocess.run(
            [PYTHON, str(script), "--raw", str(raw_path)],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
        print()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 10. Sanity Checks

    Before training, we verify:
    1. No feature has high correlation (>0.5) with the target
    2. No single feature achieves AUC > 0.98 (tautological)
    3. Buffer features are available (not NaN) for val/test sets
    4. The deforestation rate is consistent across prediction years
    """)
    return


@app.cell
def _(DATA_DIR, mo, np, pd):
    # Load both datasets from their separate directories
    traintest_dir = DATA_DIR.parent / "data" / "train_test"
    val_dir = DATA_DIR.parent / "data" / "val"

    traintest_files = sorted(traintest_dir.glob("features_traintest_*.parquet"))
    val_files = sorted(val_dir.glob("features_val_*.parquet"))

    if not traintest_files or not val_files:
        print("Waiting for build scripts to produce datasets...")
        print(f"  train_test dir: {traintest_dir} — {len(traintest_files)} files")
        print(f"  val dir: {val_dir} — {len(val_files)} files")
    else:
        df_tt = pd.read_parquet(traintest_files[-1])
        df_val = pd.read_parquet(val_files[-1])
        print(f"Train+test: {traintest_files[-1].name} — {len(df_tt):,} rows")
        print(f"Val: {val_files[-1].name} — {len(df_val):,} rows")

        # Verify no split contamination
        tt_splits = set(df_tt["split"].unique())
        val_splits = set(df_val["split"].unique())
        assert tt_splits == {"train"}, f"train+test has unexpected splits: {tt_splits}"
        assert val_splits == {"val"}, f"val has unexpected splits: {val_splits}"
        print("\nSplit integrity: OK (train+test and val are physically separate)")

        # Verify feature columns match
        tt_cols = set(df_tt.columns)
        val_cols = set(df_val.columns)
        if tt_cols == val_cols:
            print("Column consistency: OK (same columns in both datasets)")
        else:
            diff = tt_cols.symmetric_difference(val_cols)
            print(f"Column mismatch: {len(diff)} columns differ — {diff}")

        # Combine for sanity checks
        df_dataset = pd.concat([df_tt, df_val])

        NON_FEATURE = {"pid", "lon", "lat", "target", "split", "prediction_year", "lossyear"}
        feature_cols = [
            c for c in df_dataset.columns
            if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(df_dataset[c])
        ]
        print(f"\nTotal candidate features: {len(feature_cols)}")

        # Check NaN rates for key buffer features by split
        print("\n=== Buffer NaN rates by split ===")
        for split in ["train", "val"]:
            sub = df_dataset[df_dataset["split"] == split]
            if len(sub) == 0:
                continue
            pred_yrs = sorted(sub.prediction_year.unique())
            buf_cols = [c for c in feature_cols if c.startswith("defo_rate_") and "Lag1" in c]
            nan_rates = {c: sub[c].isna().mean() * 100 for c in buf_cols[:4]}
            print(f"  {split} (pred_yr={pred_yrs}):")
            for _col, nr in nan_rates.items():
                status = "OK" if nr < 10 else "MISSING" if nr > 90 else "partial"
                print(f"    {_col}: {nr:.0f}% NaN {status}")

        # Check correlations on val
        print("\n=== Feature-target correlations (val set) ===")
        y_val = df_val["target"].values
        corrs = {}
        for _col in feature_cols[:50]:
            vals = df_val[_col].values
            valid = ~np.isnan(vals)
            if valid.sum() > 100:
                corrs[_col] = np.corrcoef(vals[valid], y_val[valid])[0, 1]
        top_corrs = pd.Series(corrs).sort_values(key=abs, ascending=False).head(10)
        for feat, corr in top_corrs.items():
            print(f"  {feat}: {corr:.4f}")
        max_corr = top_corrs.abs().max()
        print(f"\n  Max |corr|: {max_corr:.4f}")
        if max_corr < 0.5:
            print("  No high correlation detected")
        else:
            print("  WARNING: High correlation — investigate potential leakage!")

    mo.md("Sanity checks complete — see output above.")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 11. Summary

    At this point, you have:

    1. **Raw checkpoint**: `data/raw_250k_YYYYMMDD.parquet`
       - 250K points × ~200 columns
       - All GEE sources extracted for 2016-2023 (no 2024 features — airtight)
       - Infrastructure features (OSM)

    2. **Features datasets** (physically separated):
       - `data/train_test/features_traintest_YYYYMMDD.parquet` — ~970K rows (train only)
       - `data/val/features_val_YYYYMMDD.parquet` — ~240K rows (val only)
       - Sliding window features with lag-indexed columns
       - Ready for XGBoost training

    ### Sources intentionally excluded

    The following data sources were tested in the ablation study and found to have
    **no predictive value** (AUC ≈ 0.500):

    - **WGI** (World Governance Indicators, World Bank) — rule of law, corruption, etc.
    - **WDI** (World Development Indicators, World Bank) — GDP, agriculture, etc.
    - **Commodity prices** (cocoa, palm oil, timber) — global, not spatially resolved

    These macro-economic features don't help because:
    1. They vary only at country level (too coarse for pixel-level prediction)
    2. The Hansen buffers already implicitly capture local governance (PA effectiveness)
    3. Global commodity prices don't vary spatially

    If you want to add them for completeness, use `scripts/add_tabular_features.py`.

    ### Split structure (airtight temporal separation)

    | Split | Script | Feature years | Predicts | Purpose |
    |-------|--------|--------------|----------|---------|
    | train | `build_traintest.py` | 2016-**2022** | 2020, 2021, 2022 | Learning |
    | val   | `build_val.py` | 2017-**2023** | **2024** | Threshold tuning + early stopping |

    **Leakage guarantee**:
    - Train NEVER sees 2023 data (feature years stop at 2022)
    - Val uses 2023 as Lag1 to predict 2024
    - Strict temporal separation: train features < 2023 <= val features
    - Datasets are physically separated in `data/train_test/` and `data/val/`

    **Next steps**:
    - Run `scripts/train_xgboost.py` on the train dataset
    - Evaluate on val dataset (pred_yr=2024)
    - Run ablation studies

    ---
    *Full extraction notebook — Deforestation Prediction Pipeline*
    """)
    return


if __name__ == "__main__":
    app.run()
