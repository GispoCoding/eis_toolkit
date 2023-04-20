from pathlib import Path
from typing import Callable, List

import rasterio
from rasterio import profiles


def raster_profile(raster_path: Path) -> profiles.Profile:
    """Get raster profile."""
    with rasterio.open(raster_path) as raster:
        profile = raster.profile
    return profile


def apply_bandwise(function: Callable, raster: rasterio.io.DatasetReader, *args, **kwargs):
    """Apply given function to band data one band at a time."""
    for band in range(1, raster.count + 1):
        band_data = raster.read(band)
        function(band_data, *args, **kwargs)


# Is this useful?
def separate_bands(raster: rasterio.io.DatasetReader):
    """TODO."""
    pass


# Is this useful?
def merge_rasters(rasters: List[rasterio.io.DatasetReader]):
    """TODO."""
    pass
