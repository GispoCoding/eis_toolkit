from pathlib import Path

import rasterio
from rasterio import profiles


def raster_profile(raster_path: Path) -> profiles.Profile:
    """Get raster profile."""
    with rasterio.open(raster_path) as raster:
        profile = raster.profile
    return profile
