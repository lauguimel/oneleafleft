"""Add country-level tabular features to an existing raw parquet checkpoint.

Features added (all assigned per country × year, then temporal lag-indexed):
  - WGI: 6 World Governance Indicators (control_corruption, gov_effectiveness,
         political_stability, rule_of_law, regulatory_quality, voice_accountability)
  - WDI: gdp_pc (GDP/capita, const. 2015 USD), pop_growth, urbanization,
         inflation, fdi, military_exp, fertility
  - Commodity prices (global, not per country):
         price_cocoa, price_palmoil, price_gold  (USD/unit, annual avg)
  - V-Dem (optional, if data/vdem_*.csv available):
         vdem_polyarchy, vdem_libdem, vdem_corr, vdem_rol
  - ACLED (optional, if data/acled_*.csv available):
         acled_events, acled_fatalities within 100 km

Country assignment: point-in-polygon with Natural Earth 110m boundaries.

Usage:
    conda activate deforest
    python scripts/add_tabular_features.py --raw data/raw_250k_20260228.parquet
    python scripts/add_tabular_features.py --raw data/raw_250k_20260228.parquet --skip-vdem --skip-acled

    # Extend to 2022-2023 for 2024 held-out test:
    python scripts/add_tabular_features.py --raw data/raw_250k_20260228.parquet --years 2016 2017 2018 2019 2020 2021 2022 2023 --skip-vdem --skip-acled
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data.gee_extraction import rebuild_features_dataset

_DEFAULT_YEARS = [2016, 2017, 2018, 2019, 2020, 2021]

# Countries present in the tile 0-10°N, 20-30°E
# Note: COG/CMR/GAB are mostly west of 20°E; SSD/SDN/TCD appear in the northeast
STUDY_COUNTRIES = ["COD", "COG", "CAF", "CMR", "GAB", "SSD", "SDN", "TCD", "UGA"]
_WB_COUNTRY_STR = ";".join(STUDY_COUNTRIES)


# ─── Commodity prices (World Bank Pink Sheet annual averages) ─────────────────
# Source: World Bank Commodity Price Data (Pink Sheet), updated monthly.
# Values are approximate annual averages. Unit: USD/metric tonne (cocoa, palmoil)
# and USD/troy oz (gold). Update from: https://www.worldbank.org/commodities
_COMMODITY_PRICES: dict[str, dict[int, float]] = {
    "price_cocoa": {         # ICCO composite price, USD/MT
        2015: 3134, 2016: 3168, 2017: 2026, 2018: 2294,
        2019: 2337, 2020: 2373, 2021: 2428, 2022: 2403, 2023: 3399,
    },
    "price_palmoil": {       # Malaysia, crude palm oil, USD/MT
        2015: 545, 2016: 637, 2017: 714, 2018: 597,
        2019: 590, 2020: 752, 2021: 1134, 2022: 1258, 2023: 907,
    },
    "price_gold": {          # LBMA London PM fix, USD/troy oz
        2015: 1160, 2016: 1249, 2017: 1257, 2018: 1268,
        2019: 1393, 2020: 1770, 2021: 1799, 2022: 1800, 2023: 1941,
    },
}


# ─── World Bank API helper ────────────────────────────────────────────────────


def _wb_api(indicator: str, country_str: str, years: list[int], timeout: int = 30) -> list[dict]:
    """Fetch a single WB API indicator for multiple countries.

    Returns list of record dicts (empty on failure).
    """
    date_range = f"{min(years) - 1}:{max(years) + 1}"
    url = (
        f"https://api.worldbank.org/v2/country/{country_str}"
        f"/indicator/{indicator}"
    )
    params = {"format": "json", "date": date_range, "per_page": 500}
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and len(data) == 2 and isinstance(data[1], list):
            return data[1]
    except Exception as exc:
        print(f"    ⚠ WB API {indicator} failed: {exc}")
    return []


def _wb_records_to_wide(
    records: list[dict], col_prefix: str, years: list[int]
) -> pd.DataFrame:
    """Pivot WB API records to wide format: iso3 × {prefix}_{year}.

    Returns DataFrame indexed by ISO3 with NaN for missing years.
    """
    data: dict[str, dict[str, float]] = {}
    for rec in records:
        iso3 = rec.get("countryiso3code", "")
        year = rec.get("date", "")
        val = rec.get("value")
        if not iso3 or val is None:
            continue
        try:
            yr_int = int(year)
        except (ValueError, TypeError):
            continue
        if yr_int not in years:
            continue
        if iso3 not in data:
            data[iso3] = {}
        data[iso3][f"{col_prefix}_{yr_int}"] = float(val)

    if not data:
        return pd.DataFrame(
            {f"{col_prefix}_{yr}": pd.Series(dtype=float) for yr in years}
        )
    df = pd.DataFrame.from_dict(data, orient="index")
    for yr in years:
        col = f"{col_prefix}_{yr}"
        if col not in df.columns:
            df[col] = np.nan
    return df


# ─── Country assignment ───────────────────────────────────────────────────────


def _load_world_boundaries(cache_dir: Path) -> gpd.GeoDataFrame:
    """Load Natural Earth 110m country boundaries, downloading if not cached."""
    import tempfile

    gpkg = cache_dir / "ne_110m_countries.gpkg"
    if gpkg.exists():
        return gpd.read_file(gpkg)

    print("  Downloading Natural Earth 110m boundaries (~400 KB)…")
    url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    # Extract to temp dir so all shapefile components (.shp, .dbf, .shx, .prj) are on disk
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(tmpdir)
        shp_files = list(Path(tmpdir).glob("*.shp"))
        if not shp_files:
            raise FileNotFoundError("No .shp file found in Natural Earth zip")
        world = gpd.read_file(str(shp_files[0]))

    # Normalise ISO3 column (can be ISO_A3, ADM0_A3, etc. depending on version)
    iso_col = next(
        (c for c in world.columns if c.upper() in ("ISO_A3", "ADM0_A3")), None
    )
    if iso_col:
        world = world.rename(columns={iso_col: "iso_a3"})

    world = world[["iso_a3", "geometry"]].copy()
    cache_dir.mkdir(parents=True, exist_ok=True)
    world.to_file(gpkg, driver="GPKG")
    print(f"  Cached to {gpkg.name}")
    return world


def _assign_countries(points: pd.DataFrame, cache_dir: Path) -> pd.Series:
    """Assign ISO3 country code to each point via point-in-polygon.

    Falls back to sjoin_nearest for points on borders / water bodies.
    Returns pd.Series indexed like points, dtype str, name='country_iso3'.
    """
    world = _load_world_boundaries(cache_dir)[["iso_a3", "geometry"]]

    pts_gdf = gpd.GeoDataFrame(
        points.copy(),
        geometry=gpd.points_from_xy(points["lon"], points["lat"]),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(pts_gdf, world, how="left", predicate="within")
    # Disambiguate if multiple polygons matched (shouldn't happen, but guard)
    joined = joined[~joined.index.duplicated(keep="first")]

    unmatched_mask = joined["iso_a3"].isna()
    n_unmatched = unmatched_mask.sum()
    if n_unmatched > 0:
        print(f"  {n_unmatched:,} points unmatched → using nearest country")
        nearest = gpd.sjoin_nearest(
            pts_gdf.loc[unmatched_mask, ["geometry"]],
            world,
            how="left",
        )
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        joined.loc[unmatched_mask, "iso_a3"] = nearest["iso_a3"]

    # Natural Earth uses non-standard codes for some countries; remap to ISO3
    _NE_REMAP = {
        "SDS": "SSD",  # South Sudan: Natural Earth → ISO 3166-1
        "-99": None,   # Disputed / no data
    }
    result = joined["iso_a3"].replace(_NE_REMAP)
    return result.rename("country_iso3")


# ─── V-Dem ────────────────────────────────────────────────────────────────────


def _load_vdem(vdem_path: Path, countries: list[str], years: list[int]) -> pd.DataFrame:
    """Load V-Dem CSV, filter to study countries, pivot to wide format.

    Expected columns: country_text_id (ISO3), year, v2x_polyarchy,
    v2x_libdem, v2x_corr, v2xcl_rol, v2x_gencs.

    Returns DataFrame indexed by ISO3 with columns like vdem_polyarchy_2019.
    """
    df = pd.read_csv(vdem_path, low_memory=False)

    # Detect ISO3 column
    iso_col = next(
        (c for c in df.columns if c in ("country_text_id", "country_iso3", "iso3")), None
    )
    if iso_col is None:
        raise ValueError(f"No ISO3 column found in V-Dem CSV. Columns: {df.columns.tolist()[:10]}")

    vdem_vars = {
        "v2x_polyarchy": "vdem_polyarchy",
        "v2x_libdem": "vdem_libdem",
        "v2x_corr": "vdem_corr",
        "v2xcl_rol": "vdem_rol",
    }
    available = {k: v for k, v in vdem_vars.items() if k in df.columns}

    df_filt = df[(df[iso_col].isin(countries)) & (df["year"].isin(years))][
        [iso_col, "year"] + list(available.keys())
    ].copy()

    wide_parts = []
    for vdem_col, prefix in available.items():
        pivot = df_filt.pivot(index=iso_col, columns="year", values=vdem_col)
        pivot.columns = [f"{prefix}_{yr}" for yr in pivot.columns]
        wide_parts.append(pivot)

    if not wide_parts:
        return pd.DataFrame(index=pd.Index(countries, name=iso_col))
    return pd.concat(wide_parts, axis=1)


# ─── ACLED ────────────────────────────────────────────────────────────────────


def _load_acled(acled_path: Path, points: pd.DataFrame, years: list[int],
                radius_km: float = 100.0) -> pd.DataFrame:
    """Compute per-point annual ACLED conflict stats within radius_km.

    Expected CSV columns: year, latitude, longitude, fatalities.
    Adds columns: acled_events_{year}, acled_fatalities_{year}.

    Uses scipy.spatial.KDTree on projected (lon/lat → km) coordinates.
    Returns DataFrame indexed like points.
    """
    from scipy.spatial import KDTree

    df_ac = pd.read_csv(acled_path, low_memory=False)
    df_ac = df_ac[df_ac["year"].isin(years)].dropna(subset=["latitude", "longitude"])

    # Approximate equirectangular projection for distance (good enough for 10° tile)
    lat_ref = points["lat"].mean()
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat_ref))

    pts_xy = np.column_stack([
        (points["lon"] - points["lon"].mean()) * km_per_deg_lon,
        (points["lat"] - points["lat"].mean()) * km_per_deg_lat,
    ])

    results: dict[str, pd.Series] = {}
    for yr in years:
        ac_yr = df_ac[df_ac["year"] == yr]
        if len(ac_yr) == 0:
            results[f"acled_events_{yr}"] = pd.Series(0, index=points.index)
            results[f"acled_fatalities_{yr}"] = pd.Series(0.0, index=points.index)
            continue

        ev_xy = np.column_stack([
            (ac_yr["longitude"].values - points["lon"].mean()) * km_per_deg_lon,
            (ac_yr["latitude"].values - points["lat"].mean()) * km_per_deg_lat,
        ])
        tree = KDTree(ev_xy)
        indices_list = tree.query_ball_point(pts_xy, r=radius_km, workers=-1)

        fat_vals = ac_yr["fatalities"].fillna(0).values
        events = np.array([len(idx) for idx in indices_list])
        fatalities = np.array([fat_vals[idx].sum() for idx in indices_list])

        results[f"acled_events_{yr}"] = pd.Series(events, index=points.index)
        results[f"acled_fatalities_{yr}"] = pd.Series(fatalities, index=points.index)

    return pd.DataFrame(results)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main(raw_path: Path, skip_vdem: bool, skip_acled: bool, years: list[int]) -> None:
    prediction_years = list(range(years[0] + 3, years[-1] + 2))
    print("=" * 60)
    print("ADD TABULAR FEATURES — WGI + WDI + Prices + (V-Dem + ACLED)")
    print("=" * 60)

    cache_dir = raw_path.parent / "boundaries"

    # ── 1. Load raw checkpoint ────────────────────────────────────────────────
    print(f"\nLoading: {raw_path.name}")
    df_raw = pd.read_parquet(raw_path)
    n_pts = len(df_raw)
    print(f"  {n_pts:,} points, {df_raw.shape[1]} columns")

    points = df_raw[["lon", "lat"]].copy()
    points.index.name = "pid"

    # ── 2. Country assignment ─────────────────────────────────────────────────
    print("\n[1] Country assignment (Natural Earth 110m)…")
    t0 = time.time()
    country_iso3 = _assign_countries(points, cache_dir)
    counts = country_iso3.value_counts()
    print(f"  Done in {time.time()-t0:.0f}s | {counts.to_dict()}")

    # Binary country dummies (for XGBoost — no label encoding needed)
    country_dummies = pd.get_dummies(country_iso3, prefix="country", dtype=float)

    # ── 3. WGI ────────────────────────────────────────────────────────────────
    print("\n[2] World Governance Indicators (WB API)…")
    wgi_indicators = {
        "CC.EST": "wgi_corruption",
        "GE.EST": "wgi_gov_effectiveness",
        "PV.EST": "wgi_political_stability",
        "RL.EST": "wgi_rule_of_law",
        "RQ.EST": "wgi_regulatory_quality",
        "VA.EST": "wgi_voice_accountability",
    }
    wgi_dfs = []
    for code, prefix in wgi_indicators.items():
        records = _wb_api(code, _WB_COUNTRY_STR, years)
        if records:
            df_ind = _wb_records_to_wide(records, prefix, years)
            wgi_dfs.append(df_ind)
            valid = df_ind.notna().sum().sum()
            print(f"  {code} ({prefix}): {valid} values for {len(df_ind)} countries")
        else:
            print(f"  {code}: no data returned")

    df_wgi_wide = pd.concat(wgi_dfs, axis=1) if wgi_dfs else pd.DataFrame(
        index=pd.Index(STUDY_COUNTRIES)
    )

    # ── 4. WDI ────────────────────────────────────────────────────────────────
    print("\n[3] World Development Indicators (WB API)…")
    wdi_indicators = {
        "NY.GDP.PCAP.KD": "gdp_pc",
        "SP.POP.GROW":     "pop_growth",
        "SP.URB.TOTL.IN.ZS": "urbanization",
        "FP.CPI.TOTL.ZG":  "inflation",
        "BX.KLT.DINV.WD.GD.ZS": "fdi",
        "MS.MIL.XPND.GD.ZS": "military_exp",
        "SP.DYN.TFRT.IN":  "fertility",
    }
    wdi_dfs = []
    for code, prefix in wdi_indicators.items():
        records = _wb_api(code, _WB_COUNTRY_STR, years)
        if records:
            df_ind = _wb_records_to_wide(records, prefix, years)
            wdi_dfs.append(df_ind)
            valid = df_ind.notna().sum().sum()
            print(f"  {code} ({prefix}): {valid} values")
        else:
            print(f"  {code}: no data returned")

    df_wdi_wide = pd.concat(wdi_dfs, axis=1) if wdi_dfs else pd.DataFrame(
        index=pd.Index(STUDY_COUNTRIES)
    )

    # Combine WGI + WDI
    df_country_features = pd.concat([df_wgi_wide, df_wdi_wide], axis=1)

    # ── 5. Commodity prices ───────────────────────────────────────────────────
    print("\n[4] Commodity prices (hardcoded annual averages — WB Pink Sheet)…")
    commodity_rows = {}
    for price_var, year_vals in _COMMODITY_PRICES.items():
        for yr in years:
            if yr in year_vals:
                commodity_rows[f"{price_var}_{yr}"] = year_vals[yr]
    df_commodity = pd.DataFrame([commodity_rows], index=["WLD"])
    print(f"  {len(commodity_rows)} price × year values loaded.")

    # ── 6. V-Dem (optional) ───────────────────────────────────────────────────
    df_vdem_wide = pd.DataFrame(index=pd.Index(STUDY_COUNTRIES))
    if not skip_vdem:
        vdem_files = sorted(raw_path.parent.glob("vdem*.csv"))
        if not vdem_files:
            vdem_files = sorted(PROJECT_DIR.glob("data/vdem*.csv"))
        if vdem_files:
            print(f"\n[5] V-Dem: loading {vdem_files[0].name}…")
            try:
                df_vdem_wide = _load_vdem(vdem_files[0], STUDY_COUNTRIES, years)
                print(f"  {df_vdem_wide.shape[1]} V-Dem columns loaded.")
            except Exception as exc:
                print(f"  ⚠ V-Dem load failed: {exc}")
        else:
            print("\n[5] V-Dem: no vdem*.csv found in data/ — skipping.")
            print("  Download from https://v-dem.net/data/the-v-dem-dataset/")
    else:
        print("\n[5] V-Dem: --skip-vdem")

    # ── 7. ACLED (optional) ───────────────────────────────────────────────────
    df_acled = pd.DataFrame(index=points.index)
    if not skip_acled:
        acled_files = sorted(raw_path.parent.glob("acled*.csv"))
        if not acled_files:
            acled_files = sorted(PROJECT_DIR.glob("data/acled*.csv"))
        if acled_files:
            print(f"\n[6] ACLED: loading {acled_files[0].name}…")
            try:
                t0 = time.time()
                df_acled = _load_acled(acled_files[0], points, years)
                print(f"  Done in {time.time()-t0:.0f}s — {df_acled.shape[1]} columns")
                ev_col = f"acled_events_{years[-1]}"
                if ev_col in df_acled.columns:
                    print(f"    {ev_col}: mean={df_acled[ev_col].mean():.2f} events/100km")
            except Exception as exc:
                print(f"  ⚠ ACLED load failed: {exc}")
        else:
            print("\n[6] ACLED: no acled*.csv found in data/ — skipping.")
            print("  Request data from https://acleddata.com/data-export-tool/")
    else:
        print("\n[6] ACLED: --skip-acled")

    # ── 8. Merge onto points ─────────────────────────────────────────────────
    print("\nMerging onto points…")

    # Merge country features (WGI + WDI + V-Dem) onto points via country_iso3
    df_country_all = pd.concat([df_country_features, df_vdem_wide], axis=1)

    # Map from country-indexed to point-indexed
    df_point_country = country_iso3.map(
        lambda iso: df_country_all.loc[iso] if iso in df_country_all.index else pd.Series(dtype=float)
    )
    if isinstance(df_point_country, pd.Series):
        # Edge case: all NaN
        df_point_country = pd.DataFrame(index=points.index)
    else:
        df_point_country = pd.DataFrame(
            df_country_all.reindex(country_iso3).values,
            index=points.index,
            columns=df_country_all.columns,
        )

    # Map commodity prices onto points (same for all points)
    commodity_cols = list(df_commodity.columns)
    for col in commodity_cols:
        df_point_country[col] = float(df_commodity[col].iloc[0])

    # ── 9. Patch raw parquet ──────────────────────────────────────────────────
    print("\nPatching raw parquet…")
    # All new columns
    new_static_cols = ["country_iso3"] + list(country_dummies.columns)
    new_annual_cols = list(df_point_country.columns) + list(df_acled.columns)
    all_new_cols = new_static_cols + new_annual_cols

    df_raw_patched = df_raw.drop(columns=[c for c in all_new_cols if c in df_raw.columns])
    df_raw_patched = df_raw_patched.join(
        pd.DataFrame({"country_iso3": country_iso3}).join(country_dummies)
    )
    if len(df_point_country.columns) > 0:
        df_raw_patched = df_raw_patched.join(df_point_country)
    if len(df_acled.columns) > 0:
        df_raw_patched = df_raw_patched.join(df_acled)

    df_raw_patched.to_parquet(raw_path)
    print(f"  Saved: {raw_path.name} ({df_raw_patched.shape[1]} columns)")

    # ── 10. Rebuild features dataset ──────────────────────────────────────────
    print("\nRebuilding features dataset…")
    df_dataset = rebuild_features_dataset(df_raw_patched, years, prediction_years)
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
    print("TABULAR FEATURES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add country-level tabular features to raw parquet")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--skip-vdem", action="store_true", help="Skip V-Dem loading")
    parser.add_argument("--skip-acled", action="store_true", help="Skip ACLED loading")
    parser.add_argument("--years", type=int, nargs="+", default=_DEFAULT_YEARS,
                        help="Years to fetch WGI/WDI/prices for (default: 2016-2021). "
                             "To add 2022-2023: --years 2016 2017 2018 2019 2020 2021 2022 2023")
    args = parser.parse_args()
    main(args.raw, args.skip_vdem, args.skip_acled, sorted(args.years))
