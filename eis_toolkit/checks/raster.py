from typing import List, Sequence

import rasterio


def check_matching_cell_size(  # type: ignore[no-any-unimported]
    rasters: List[rasterio.io.DatasetReader],
) -> bool:
    """
    Check if every raster in a list has matching cell size.

    Args:
        rasters: List of rasters to check.

    Returns:
        Bool: True if cell size of each raster matches, False if not.
    """

    pixel_size = [rasters[0].transform.a, rasters[0].transform.e]
    for raster in rasters:
        if [raster.transform.a, raster.transform.e] != pixel_size:
            return False
    return True


def check_matching_pixel_alignment(  # type: ignore[no-any-unimported]
    rasters: List[rasterio.io.DatasetReader],
) -> bool:
    """
    Check if every raster in a list has matching cell size and matching pixel alignment.

    Args:
        rasters: List of rasters to check.

    Returns:
        bool: True if cell size and pixel alignment matches, False if not.
    """

    if check_matching_cell_size(rasters):
        pixel_size_x, pixel_size_y = rasters[0].transform.a, abs(rasters[0].transform.e)
        left_pixel, top_pixel = rasters[0].bounds.left, rasters[0].bounds.top
        for raster in rasters:
            if (
                abs(left_pixel - raster.bounds.left) % pixel_size_x != 0
                or abs(top_pixel - raster.bounds.top) % pixel_size_y != 0
            ):
                return False
        return True
    else:
        return False


def check_matching_bounds(  # type: ignore[no-any-unimported]
    rasters: List[rasterio.io.DatasetReader],
) -> bool:
    """
    Check if every raster in a list has matching bounds.

    Args:
        rasters: List of rasters to check.

    Returns:
        Bool: True if bounds of each raster matches, False if not.
    """

    bounds = rasters[0].bounds
    for raster in rasters:
        if raster.bounds != bounds:
            return False
    return True


def check_raster_bands(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    bands: Sequence[int],
) -> bool:
    """
    Check if selection of bands is contained in the raster.

    Args:
        raster: Raster to be checked.

    Returns:
        Bool: True if all bands exist, False if not.
    """
    return all(band in range(1, raster.count + 1) for band in bands)
