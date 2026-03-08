"""Enrich WDPA protected area features beyond binary in/out + distance.

Adds:
  1. IUCN category (one-hot: strict Ia/Ib, II-IV, V-VI, not_reported)
  2. Distance to nearest PA boundary (not centroid) — frontier effect
  3. Deforestation effectiveness: defo rate INSIDE each PA (proxy for governance)
  4. Pressure buffer: defo rate in 0-5km ring around each PA

Usage:
    conda activate deforest
    python scripts/enrich_wdpa.py --raw data/raw_250k_20260228.parquet
    python scripts/enrich_wdpa.py --raw data/raw_250k_20260228.parquet --skip-rebuild
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data.gee_utils import init_gee
from data.gee_extraction import rebuild_features_dataset


def extract_iucn_category(points: pd.DataFrame) -> pd.DataFrame:
    """Extract IUCN category for each point from WDPA polygons via GEE.

    Returns one-hot columns:
      - iucn_strict (Ia, Ib)
      - iucn_moderate (II, III, IV)
      - iucn_sustainable (V, VI)
      - iucn_not_reported (Not Reported, Not Applicable, Not Assigned)
    """
    import ee

    wdpa = ee.FeatureCollection("WCMC/WDPA/current/polygons")

    # IUCN category groups
    strict = wdpa.filter(ee.Filter.inList("IUCN_CAT", ["Ia", "Ib"]))
    moderate = wdpa.filter(ee.Filter.inList("IUCN_CAT", ["II", "III", "IV"]))
    sustainable = wdpa.filter(ee.Filter.inList("IUCN_CAT", ["V", "VI"]))

    # Paint each group onto an image
    base = ee.Image(0).byte()
    iucn_strict = base.paint(featureCollection=strict, color=1).rename("iucn_strict")
    iucn_moderate = base.paint(featureCollection=moderate, color=1).rename("iucn_moderate")
    iucn_sustainable = base.paint(featureCollection=sustainable, color=1).rename("iucn_sustainable")

    # "Not reported" = in_protected but not in any of the 3 groups
    all_pa = base.paint(featureCollection=wdpa, color=1)
    classified = iucn_strict.Or(iucn_moderate).Or(iucn_sustainable)
    iucn_not_reported = all_pa.And(classified.Not()).rename("iucn_not_reported")

    multi = ee.Image.cat([iucn_strict, iucn_moderate, iucn_sustainable, iucn_not_reported])

    # Extract at point locations
    from data.gee_extraction import extract_image
    return extract_image(multi, points, scale=1000)


def extract_boundary_distance(points: pd.DataFrame) -> pd.DataFrame:
    """Distance to nearest PA boundary (edge), regardless of in/out.

    For points inside PAs: distance to the nearest edge (smaller = more frontier).
    For points outside PAs: distance to nearest PA boundary.
    Capped at 100 km. In km.
    """
    import ee

    wdpa = ee.FeatureCollection("WCMC/WDPA/current/polygons")
    pa_mask = ee.Image(0).byte().paint(featureCollection=wdpa, color=1)

    # PA boundary = edge pixels (inside PA, adjacent to outside)
    # Using Canny edge detection on the PA mask
    pa_edge = pa_mask.toFloat().convolve(ee.Kernel.laplacian8()).abs().gt(0)

    # Distance from each pixel to nearest edge pixel
    dist_boundary = pa_edge.fastDistanceTransform(256).sqrt() \
        .multiply(ee.Image.pixelArea().sqrt()) \
        .divide(1000.0) \
        .rename("dist_pa_boundary_km")

    from data.gee_extraction import extract_image
    return extract_image(dist_boundary, points, scale=1000)


def extract_pa_effectiveness(points: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Deforestation rate INSIDE PAs and in 5km buffer ring around PAs.

    Produces two static features:
      - pa_defo_rate: mean deforestation rate inside the nearest PA (proxy for effectiveness)
      - pa_pressure_ring: mean deforestation rate in 0-5km ring around PAs (external pressure)

    Uses Hansen lossyear over the specified years.
    """
    import ee

    hansen = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
    wdpa = ee.FeatureCollection("WCMC/WDPA/current/polygons")

    pa_mask = ee.Image(0).byte().paint(featureCollection=wdpa, color=1)

    # Total deforestation in recent years (union of all loss years)
    yr_codes = [yr - 2000 for yr in years]
    loss_any = hansen.select("lossyear").unmask(0)
    loss_recent = ee.Image(0)
    for yc in yr_codes:
        loss_recent = loss_recent.Or(loss_any.eq(yc))
    loss_recent = loss_recent.toFloat()

    # Deforestation rate inside PAs (smoothed over 5km for spatial context)
    pa_defo = loss_recent.updateMask(pa_mask).unmask(0)
    pa_defo_smooth = pa_defo.reduceNeighborhood(
        reducer=ee.Reducer.mean(),
        kernel=ee.Kernel.circle(5000, "meters"),
    ).rename("pa_defo_rate")

    # 5km pressure ring: outside PA but within 5km
    # Distance to PA boundary (outside only)
    dist_outside = pa_mask.Not().toFloat().cumulativeCost(
        source=pa_mask, maxDistance=5000
    )
    ring_mask = dist_outside.lte(5000).And(pa_mask.Not())
    ring_defo = loss_recent.updateMask(ring_mask).unmask(0)
    ring_defo_smooth = ring_defo.reduceNeighborhood(
        reducer=ee.Reducer.mean(),
        kernel=ee.Kernel.circle(5000, "meters"),
    ).rename("pa_pressure_ring")

    multi = ee.Image.cat([pa_defo_smooth, ring_defo_smooth])

    from data.gee_extraction import extract_image
    return extract_image(multi, points, scale=1000)


def main(raw_path: Path, skip_rebuild: bool = False):
    print("=" * 60)
    print("ENRICH WDPA — Protected Area Effectiveness Features")
    print("=" * 60)

    # Load raw
    print(f"\nLoading: {raw_path.name}")
    df_raw = pd.read_parquet(raw_path)
    print(f"  {len(df_raw):,} points, {df_raw.shape[1]} columns")

    points = df_raw[["lon", "lat"]].copy()
    points.index.name = "pid"

    # Init GEE
    print("\nInitializing GEE...")
    init_gee()
    print("  OK")

    years = [2016, 2017, 2018, 2019, 2020, 2021]
    new_cols = []

    # ── 1. IUCN categories ──────────────────────────────────────────────────
    print("\n[1] IUCN category extraction...")
    t0 = time.time()
    try:
        df_iucn = extract_iucn_category(points)
        print(f"  Done in {time.time()-t0:.0f}s")
        for col in df_iucn.columns:
            pct = df_iucn[col].mean() * 100
            print(f"    {col}: {pct:.1f}% of points")
        new_cols.extend(df_iucn.columns.tolist())
    except Exception as e:
        print(f"  FAILED: {e}")
        df_iucn = None

    # ── 2. Distance to PA boundary ──────────────────────────────────────────
    print("\n[2] Distance to PA boundary (frontier effect)...")
    t0 = time.time()
    try:
        df_boundary = extract_boundary_distance(points)
        print(f"  Done in {time.time()-t0:.0f}s")
        vals = df_boundary["dist_pa_boundary_km"].dropna()
        print(f"    median={vals.median():.1f}km, mean={vals.mean():.1f}km, "
              f"<1km={float((vals < 1).mean()*100):.1f}%")
        new_cols.extend(df_boundary.columns.tolist())
    except Exception as e:
        print(f"  FAILED: {e}")
        df_boundary = None

    # ── 3. PA effectiveness + pressure ring ─────────────────────────────────
    print("\n[3] PA deforestation effectiveness + pressure ring...")
    t0 = time.time()
    try:
        df_effect = extract_pa_effectiveness(points, years)
        print(f"  Done in {time.time()-t0:.0f}s")
        for col in df_effect.columns:
            vals = df_effect[col].dropna()
            print(f"    {col}: mean={vals.mean():.4f}, nonzero={float((vals > 0).mean()*100):.1f}%")
        new_cols.extend(df_effect.columns.tolist())
    except Exception as e:
        print(f"  FAILED: {e}")
        df_effect = None

    # ── 4. Patch raw parquet ────────────────────────────────────────────────
    print("\nPatching raw parquet...")
    df_raw_patched = df_raw.drop(columns=[c for c in new_cols if c in df_raw.columns])
    for df_part in [df_iucn, df_boundary, df_effect]:
        if df_part is not None:
            df_raw_patched = df_raw_patched.join(df_part)

    df_raw_patched.to_parquet(raw_path)
    print(f"  Saved: {raw_path.name} ({df_raw_patched.shape[1]} columns)")

    # ── 5. Rebuild features dataset ─────────────────────────────────────────
    if not skip_rebuild:
        import re as _re
        all_years = sorted({
            int(m.group(1))
            for c in df_raw_patched.columns
            for m in [_re.search(r"_(\d{4})$", c)]
            if m and 2000 <= int(m.group(1)) <= 2030
        })
        rebuild_pred_years = list(range(all_years[0] + 3, all_years[-1] + 2))
        print(f"\nRebuilding features dataset...")
        print(f"  detected years={all_years}, prediction_years={rebuild_pred_years}")
        df_dataset = rebuild_features_dataset(df_raw_patched, all_years, rebuild_pred_years)
        print(f"  Dataset shape: {df_dataset.shape}")

        for split in ["train", "val", "test"]:
            sub = df_dataset[df_dataset["split"] == split]
            pos_rate = sub["target"].mean() * 100 if len(sub) > 0 else 0
            print(f"  {split:5s}: {len(sub):>7,} rows — {pos_rate:.1f}% deforested")

        stem = raw_path.stem.replace("raw_", "features_")
        out_path = raw_path.parent / f"{stem}.parquet"
        df_dataset.to_parquet(out_path, index=False)
        print(f"  Saved: {out_path.name}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("WDPA ENRICHMENT COMPLETE")
    print(f"  New columns: {new_cols}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich WDPA features")
    parser.add_argument("--raw", type=Path, required=True, help="Path to raw_*.parquet")
    parser.add_argument("--skip-rebuild", action="store_true")
    args = parser.parse_args()
    main(args.raw, args.skip_rebuild)
