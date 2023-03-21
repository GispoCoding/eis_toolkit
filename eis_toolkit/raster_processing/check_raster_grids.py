from typing import List

import rasterio

from eis_toolkit.checks.crs import check_matching_crs
from eis_toolkit.checks.raster_checks import check_matching_bounds, check_matching_pixel_alignment


def _check_raster_grids(  # type: ignore[no-any-unimported]
    rasters: List[rasterio.io.DatasetReader],
    same_extent: bool = False,
) -> bool:

    if not check_matching_crs(rasters):
        return False
    if not check_matching_pixel_alignment(rasters):
        return False
    if same_extent and not check_matching_bounds(rasters):
        return False
    return True


def check_raster_grids(  # type: ignore[no-any-unimported]
    rasters: List[rasterio.io.DatasetReader], same_extent: bool = False
) -> bool:
    """
    Check the set of input rasters for matching gridding and optionally matching bounds.

    Args:
        rasters: List of rasters to test for matching gridding.
        same_extent: optional boolean argument that determines if rasters are tested for matching bounds.
            Default set to False.

    Returns:
        True if gridding and optionally bounds matches, False if not.
    """
    check = _check_raster_grids(rasters=rasters, same_extent=same_extent)
    return check
