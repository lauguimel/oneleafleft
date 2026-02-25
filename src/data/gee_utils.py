"""Google Earth Engine initialization utilities."""

import os
from pathlib import Path

import ee

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_PROJECT = "ee-guillaumemaitrejean"


def init_gee(project: str | None = None) -> None:
    """Initialize GEE with project ID.

    Resolution order:
      1. Explicit ``project`` argument
      2. EARTHENGINE_PROJECT environment variable
      3. Default project constant
    """
    project = project or os.environ.get("EARTHENGINE_PROJECT") or _DEFAULT_PROJECT

    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
