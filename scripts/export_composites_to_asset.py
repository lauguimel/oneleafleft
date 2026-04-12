"""Export HLS annual composites as GEE Assets (not Drive).

Stores pre-computed composites inside GEE's infrastructure so that
subsequent computePixels calls are fast (pixel reading, no computation).

This replaces the export_tiles_gee.py approach (which exported to Drive
and produced 300+ GB of data). Assets stay in GEE — no download needed.

Then extract_chips_fast.py reads chips from these assets via computePixels.

Usage:
    conda run -n deforest python scripts/export_composites_to_asset.py
    conda run -n deforest python scripts/export_composites_to_asset.py --check
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

# ── Config ────────────────────────────────────────────────────────────────────
STUDY_AREA = {"lon_min": 20.0, "lon_max": 30.0, "lat_min": 0.0, "lat_max": 10.0}
ASSET_FOLDER = "projects/ee-guillaumemaitrejean/assets/deforest"
HLS_COLLECTION = "NASA/HLS/HLSL30/v002"
HLS_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7"]
YEARS = list(range(2014, 2022))  # 2014–2021
EXPORT_SCALE = 30
CRS = "EPSG:4326"


def export_composites():
    import ee
    from data.gee_utils import init_gee
    init_gee()

    aoi = ee.Geometry.Rectangle([
        STUDY_AREA["lon_min"], STUDY_AREA["lat_min"],
        STUDY_AREA["lon_max"], STUDY_AREA["lat_max"],
    ])

    # Create asset folder if needed
    try:
        ee.data.createAsset({"type": "FOLDER"}, ASSET_FOLDER)
        print(f"Created asset folder: {ASSET_FOLDER}")
    except Exception:
        print(f"Asset folder exists: {ASSET_FOLDER}")

    tasks = []

    for year in YEARS:
        asset_id = f"{ASSET_FOLDER}/hls_{year}"
        print(f"Submitting HLS composite {year} → {asset_id}")

        col = (
            ee.ImageCollection(HLS_COLLECTION)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .filterBounds(aoi)
        )

        def mask_clouds(img):
            fmask = img.select("Fmask")
            clear = fmask.bitwiseAnd(0b11110).eq(0)
            return img.updateMask(clear)

        composite = col.map(mask_clouds).select(HLS_BANDS).median().toFloat()
        ndvi = composite.normalizedDifference(["B5", "B4"]).rename("NDVI")
        nbr = composite.normalizedDifference(["B5", "B7"]).rename("NBR")
        composite = composite.addBands(ndvi).addBands(nbr)

        task = ee.batch.Export.image.toAsset(
            image=composite,
            description=f"hls_{year}_asset",
            assetId=asset_id,
            region=aoi,
            scale=EXPORT_SCALE,
            crs=CRS,
            maxPixels=1e10,
        )
        task.start()
        tasks.append((f"hls_{year}", task))

    print(f"\nSubmitted {len(tasks)} exports to GEE Assets")
    print(f"Asset folder: {ASSET_FOLDER}")
    print(f"Monitor: https://code.earthengine.google.com/tasks")
    print(f"\nOnce done, run: python scripts/extract_chips_fast.py")


def check_status():
    import ee
    from data.gee_utils import init_gee
    init_gee()

    print(f"Asset folder: {ASSET_FOLDER}")
    try:
        assets = ee.data.listAssets({"parent": ASSET_FOLDER})
        existing = [a["id"].split("/")[-1] for a in assets.get("assets", [])]
        print(f"Assets present: {existing}")
    except Exception as e:
        print(f"Could not list assets: {e}")

    tasks = ee.batch.Task.list()
    for t in tasks[:20]:
        desc = t.config.get("description", "?")
        if "asset" in desc or "hls_" in desc:
            print(f"  {desc:30s} {t.state}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        check_status()
    else:
        export_composites()


if __name__ == "__main__":
    main()
