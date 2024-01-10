import rasterio
from beartype import beartype
from beartype.typing import Sequence, Union

from eis_toolkit.utilities.checks.crs import check_matching_crs


@beartype
def check_matching_cell_size(
    rasters: Sequence[rasterio.io.DatasetReader],
) -> bool:
    """Check if all input rasters have matching cell size.

    Args:
        rasters: List of rasters to check.

    Returns:
        True if cell size of each raster matches, False if not.
    """

    pixel_size = [rasters[0].transform.a, rasters[0].transform.e]
    for raster in rasters:
        if [raster.transform.a, raster.transform.e] != pixel_size:
            return False
    return True


@beartype
def check_matching_pixel_alignment(
    rasters: Sequence[rasterio.io.DatasetReader],
) -> bool:
    """Check if all input rasters have matching cell size and matching pixel alignment.

    Args:
        rasters: List of rasters to check.

    Returns:
        True if cell size and pixel alignment matches, False if not.
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


@beartype
def check_matching_bounds(
    rasters: Sequence[rasterio.io.DatasetReader],
) -> bool:
    """Check if all input rasters have matching bounds.

    Args:
        rasters: List of rasters to check.

    Returns:
        True if bounds of each raster matches, False if not.
    """

    bounds = rasters[0].bounds
    for raster in rasters:
        if raster.bounds != bounds:
            return False
    return True


@beartype
def check_raster_grids(
    rasters: Sequence[Union[rasterio.io.DatasetReader, rasterio.io.DatasetWriter]], same_extent: bool = False
) -> bool:
    """
    Check all input rasters for matching gridding and optionally matching bounds.

    Args:
        rasters: List of rasters to test for matching gridding.
        same_extent: Optional boolean argument that determines if rasters are tested for matching bounds.
            Default set to False.

    Returns:
        True if gridding and optionally bounds matches, False if not.
    """
    if not check_matching_crs(rasters):
        return False
    if not check_matching_pixel_alignment(rasters):
        return False
    if same_extent and not check_matching_bounds(rasters):
        return False
    return True


@beartype
def check_raster_bands(raster: rasterio.io.DatasetReader, bands: Sequence[int]) -> bool:
    """Check if selection of bands is contained in the raster.

    Args:
        raster: Raster to be checked.

    Returns:
        True if all bands exist, False if not.
    """
    return all(band in range(1, raster.count + 1) for band in bands)


@beartype
def check_quadratic_pixels(raster: rasterio.io.DatasetReader) -> bool:
    """
    Check if raster pixels are quadratic.

    Args:
        raster: Raster to be checked.

    Returns:
        True if pixels are quadratic, False if not.
    """
    if raster.res[0] == raster.res[1]:
        return True
    else:
        return False
