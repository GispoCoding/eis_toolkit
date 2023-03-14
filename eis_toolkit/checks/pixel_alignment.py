from typing import List
from eis_toolkit.checks.cell_size import check_matching_cell_size

import rasterio


def check_matching_pixel_alignment(
    rasters: List[rasterio.io.DatasetReader],
) -> bool:
    """Check if every raster in a list has matching cell size and matching pixel alignment.

    Args:
        rasters: List of rasters to check.

    Returns:
        bool: True if cell size and pixel alignment matches, false if not.
    """

    if check_matching_cell_size(rasters):
        pixel_size_x, pixel_size_y = rasters[0].transform.a, abs(rasters[0].transform.e)
        left_pixel, top_pixel = rasters[0].bounds.left, rasters[0].bounds.top
        for raster in rasters:
            if (left_pixel - raster.bounds.left == 0 and top_pixel - raster.bounds.top == 0 or
                left_pixel - raster.bounds.left == 0 and (top_pixel - raster.bounds.top) % pixel_size_y == 0 or
                    (left_pixel - raster.bounds.left) % pixel_size_x == 0 and top_pixel - raster.bounds.top == 0):
                return True
            if abs(left_pixel - raster.bounds.left) % pixel_size_x != 0 or abs(
                    top_pixel - raster.bounds.top) % pixel_size_y != 0:
                return False
        return True
    else:
        return False
