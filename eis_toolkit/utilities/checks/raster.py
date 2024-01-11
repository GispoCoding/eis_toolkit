import rasterio
from beartype import beartype
from beartype.typing import Sequence

from eis_toolkit.utilities.checks.crs import check_matching_crs


@beartype
def check_matching_cell_size(
    raster_profiles: Sequence[rasterio.profiles.Profile],
) -> bool:
    """Check from profiles of rasters if all have matching cell size.

    Args:
        rasters: List of profiles of rasters to check.

    Returns:
        True if cell size of each raster matches, False if not.
    """

    pixel_size = [raster_profiles[0]["transform"][0], raster_profiles[0]["transform"][4]]
    for raster_profile in raster_profiles:
        if [raster_profile["transform"][0], raster_profile["transform"][4]] != pixel_size:
            return False
    return True


@beartype
def check_matching_pixel_alignment(
    raster_profiles: Sequence[rasterio.profiles.Profile],
) -> bool:
    """Check from profiles of rasters if all have matching cell size and matching pixel alignment.

    Args:
        rasters: List of profiles of rasters to check.

    Returns:
        True if cell size and pixel alignment matches, False if not.
    """

    if check_matching_cell_size(raster_profiles):
        pixel_size_x, pixel_size_y = raster_profiles[0]["transform"][0], abs(raster_profiles[0]["transform"][4])
        left_pixel, top_pixel = raster_profiles[0]["transform"][2], raster_profiles[0]["transform"][5]
        for raster_profile in raster_profiles:
            if (
                abs(left_pixel - raster_profile["transform"][2]) % pixel_size_x != 0
                or abs(top_pixel - raster_profile["transform"][5]) % pixel_size_y != 0
            ):
                return False
        return True
    else:
        return False


@beartype
def check_matching_bounds(
    raster_profiles: Sequence[rasterio.profiles.Profile],
) -> bool:
    """Check from profiles if all rasters have matching bounds.

    Args:
        rasters: List of profiles of rasters to check.

    Returns:
        True if bounds of each raster matches, False if not.
    """

    bounds = (raster_profiles[0]["transform"][2], raster_profiles[0]["transform"][5])
    for raster_profile in raster_profiles:
        if (raster_profile["transform"][2], raster_profile["transform"][5]) != bounds:
            return False
    return True


@beartype
def check_raster_grids(raster_profiles: Sequence[rasterio.profiles.Profile], same_extent: bool = False) -> bool:
    """
    Check from profiles of rasters for matching gridding and optionally matching bounds.

    Args:
        rasters: List of profiles of rasters to test for matching gridding.
        same_extent: Optional boolean argument that determines if rasters are tested for matching bounds.
            Default set to False.

    Returns:
        True if gridding and optionally bounds matches, False if not.
    """
    if not check_matching_crs(raster_profiles):
        return False
    if not check_matching_pixel_alignment(raster_profiles):
        return False
    if same_extent and not check_matching_bounds(raster_profiles):
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
