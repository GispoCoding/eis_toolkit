from typing import List

import rasterio


def check_matching_cell_size(
    rasters: List[rasterio.io.DatasetReader],
) -> bool:
    """Check if every raster in a list has matching cell size.

    Args:
        rasters: List of rasters to check.

    Returns:
        bool: True if cell size of each raster matches, false if not.
    """

    pixel_size = [rasters[0].transform.a, rasters[0].transform.e]
    for raster in rasters:
        if [raster.transform.a, raster.transform.e] != pixel_size:
            return False
    return True
