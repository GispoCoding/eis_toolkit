from typing import List

import rasterio

from eis_toolkit.checks.crs import check_matching_crs
from eis_toolkit.checks.raster_checks import check_matching_bounds, check_matching_pixel_alignment


def _gridding_check(
    rasters: List[rasterio.io.DatasetReader],
    same_extent: bool = False,
) -> bool:  # type: ignore[no-any-unimported]

    if not check_matching_crs(rasters):
        return False
    if not check_matching_pixel_alignment(rasters):
        return False
    if same_extent and not check_matching_bounds(rasters):
        return False
    return True


def gridding_check(
    rasters: List[rasterio.io.DatasetReader], same_extent: bool = False
) -> bool:  # type: ignore[no-any-unimported]
    """
    Check the set of input rasters for matching gridding and optionally matching bounds.

    Args:
        rasters: List of rasters to test for matching gridding.
        same_extent: optional boolean argument that determines if rasters are tested for matching bounds.
        Default set to False.

    Returns:
        True if gridding and optionally bounds matches, False if not.
    """
    check = _gridding_check(rasters=rasters, same_extent=same_extent)
    return check
