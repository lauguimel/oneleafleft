"""Upload forecast GeoTIFFs to Google Earth Engine as assets.

Uploads COG files to the user's GEE asset folder so they can be served
by the GEE App without any external hosting.

Prerequisites:
  - earthengine CLI installed: pip install earthengine-api
  - Authenticated: earthengine authenticate
  - GCS bucket for staging (or use --via-drive for Drive upload)

Usage:
    # Upload a single forecast
    conda run -n deforest python scripts/upload_gee_asset.py \
        results/forecast_2024.tif \
        --asset-id users/guillaumemaitrejean/deforest/forecast_2024

    # Upload all forecasts
    conda run -n deforest python scripts/upload_gee_asset.py \
        results/forecast_2024.tif results/forecast_2025.tif results/forecast_2026.tif \
        --asset-prefix users/guillaumemaitrejean/deforest/forecast
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def upload_to_gee(
    tif_path: Path,
    asset_id: str,
    nodata: float = -1.0,
):
    """Upload a GeoTIFF to GEE as an image asset."""
    print(f"Uploading {tif_path.name} → {asset_id}")

    cmd = [
        "earthengine", "upload", "image",
        f"--asset_id={asset_id}",
        f"--nodata_value={nodata}",
        str(tif_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        return False

    print(f"  Task started: {result.stdout.strip()}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", type=Path, nargs="+", help="GeoTIFF files to upload")
    parser.add_argument("--asset-id", type=str, default=None,
                        help="Full asset ID (for single file)")
    parser.add_argument("--asset-prefix", type=str,
                        default="users/guillaumemaitrejean/deforest/forecast",
                        help="Asset ID prefix (for multiple files)")
    args = parser.parse_args()

    for tif_path in args.files:
        if not tif_path.exists():
            print(f"SKIP: {tif_path} not found")
            continue

        if args.asset_id and len(args.files) == 1:
            asset_id = args.asset_id
        else:
            # Derive asset ID from filename: forecast_2024.tif → prefix_2024
            stem = tif_path.stem
            asset_id = f"{args.asset_prefix}_{stem.split('_')[-1]}"

        upload_to_gee(tif_path, asset_id)

    print("\nMonitor uploads at: https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    main()
