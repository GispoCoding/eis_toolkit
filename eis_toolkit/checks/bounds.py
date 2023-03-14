from typing import List

import rasterio


def check_matching_bounds(
    rasters: List[rasterio.io.DatasetReader],
) -> bool:
    """Check if every raster in a list has matching bounds.

    Args:
        rasters: List of rasters to check.

    Returns:
        bool: True if bounds of each raster matches, false if not.
    """

    bounds = rasters[0].bounds
    for raster in rasters:
        if raster.bounds != bounds:
            return False
    return True
