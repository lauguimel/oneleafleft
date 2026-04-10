"""Export full-extent GeoTIFFs from GEE for local chipping.

Instead of extracting 20K chips one-by-one via computePixels (~14 days),
export full annual composites + static layers as GeoTIFFs to Google Drive,
then chip locally with rasterio (minutes).

Exports:
  - HLS annual median composites (6 bands + NDVI + NBR) for years 2014-2021
  - SRTM (elevation + slope)
  - Hansen treecover2000
  - Hansen lossyear

Study area: 20°E-30°E, 0°N-10°N (the 250K points bounding box)

Output: Google Drive folder 'deforest_tiles/' → download locally to data/tiles/

Usage:
    conda run -n deforest python scripts/export_tiles_gee.py
    conda run -n deforest python scripts/export_tiles_gee.py --check  # check export status
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

# ── Config ────────────────────────────────────────────────────────────────────
STUDY_AREA = {
    "lon_min": 20.0, "lon_max": 30.0,
    "lat_min": 0.0, "lat_max": 10.0,
}
EXPORT_SCALE = 30  # metres
CRS = "EPSG:4326"
DRIVE_FOLDER = "deforest_tiles"

HLS_COLLECTION = "NASA/HLS/HLSL30/v002"
HLS_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7"]
HANSEN_ASSET = "UMD/hansen/global_forest_change_2024_v1_12"
SRTM_ASSET = "USGS/SRTMGL1_003"

# Years needed: feature windows for pred_years 2019-2022 with T=5
# pred_year=2019: features 2014-2018
# pred_year=2022: features 2017-2021
YEARS = list(range(2014, 2022))  # 2014, 2015, ..., 2021


def export_tiles():
    import ee
    from data.gee_utils import init_gee

    init_gee()

    aoi = ee.Geometry.Rectangle([
        STUDY_AREA["lon_min"], STUDY_AREA["lat_min"],
        STUDY_AREA["lon_max"], STUDY_AREA["lat_max"],
    ])

    tasks = []

    # ── HLS annual composites ─────────────────────────────────────────────
    for year in YEARS:
        print(f"Submitting HLS composite {year}...")

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

        task = ee.batch.Export.image.toDrive(
            image=composite,
            description=f"hls_{year}",
            folder=DRIVE_FOLDER,
            fileNamePrefix=f"hls_{year}",
            region=aoi,
            scale=EXPORT_SCALE,
            crs=CRS,
            maxPixels=1e10,
            fileFormat="GeoTIFF",
        )
        task.start()
        tasks.append((f"hls_{year}", task))

    # ── Static layers ─────────────────────────────────────────────────────
    hansen = ee.Image(HANSEN_ASSET)

    # Lossyear (need full range for masking)
    print("Submitting Hansen lossyear...")
    task_ly = ee.batch.Export.image.toDrive(
        image=hansen.select("lossyear").toByte(),
        description="hansen_lossyear",
        folder=DRIVE_FOLDER,
        fileNamePrefix="hansen_lossyear",
        region=aoi,
        scale=EXPORT_SCALE,
        crs=CRS,
        maxPixels=1e10,
        fileFormat="GeoTIFF",
    )
    task_ly.start()
    tasks.append(("hansen_lossyear", task_ly))

    # Treecover2000
    print("Submitting Hansen treecover2000...")
    task_tc = ee.batch.Export.image.toDrive(
        image=hansen.select("treecover2000").toByte(),
        description="hansen_treecover2000",
        folder=DRIVE_FOLDER,
        fileNamePrefix="hansen_treecover2000",
        region=aoi,
        scale=EXPORT_SCALE,
        crs=CRS,
        maxPixels=1e10,
        fileFormat="GeoTIFF",
    )
    task_tc.start()
    tasks.append(("hansen_treecover2000", task_tc))

    # SRTM elevation + slope
    print("Submitting SRTM...")
    srtm = ee.Image(SRTM_ASSET)
    elevation = srtm.select("elevation").toFloat()
    slope = ee.Terrain.slope(srtm).toFloat()
    srtm_stack = elevation.addBands(slope.rename("slope"))

    task_srtm = ee.batch.Export.image.toDrive(
        image=srtm_stack,
        description="srtm",
        folder=DRIVE_FOLDER,
        fileNamePrefix="srtm",
        region=aoi,
        scale=EXPORT_SCALE,
        crs=CRS,
        maxPixels=1e10,
        fileFormat="GeoTIFF",
    )
    task_srtm.start()
    tasks.append(("srtm", task_srtm))

    print(f"\n{'='*50}")
    print(f"Submitted {len(tasks)} export tasks to Google Drive folder '{DRIVE_FOLDER}'")
    print(f"Monitor at: https://code.earthengine.google.com/tasks")
    print(f"Or run: python scripts/export_tiles_gee.py --check")
    print(f"\nOnce done, download to data/tiles/ and run scripts/chip_from_tiles.py")

    return tasks


def check_status():
    import ee
    from data.gee_utils import init_gee
    init_gee()

    tasks = ee.batch.Task.list()
    active = [t for t in tasks if t.state in ("READY", "RUNNING")]
    completed = [t for t in tasks if t.state == "COMPLETED"
                 and "deforest_tiles" in str(t.config)]
    failed = [t for t in tasks if t.state == "FAILED"]

    print(f"Active:    {len(active)}")
    print(f"Completed: {len(completed)}")
    print(f"Failed:    {len(failed)}")

    for t in tasks[:15]:
        desc = t.config.get("description", "?")
        print(f"  {desc:30s} {t.state}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Check export status")
    args = parser.parse_args()

    if args.check:
        check_status()
    else:
        export_tiles()


if __name__ == "__main__":
    main()
