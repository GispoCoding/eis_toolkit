import numpy as np
import rasterio
import rasterio.profiles
from beartype import beartype
from beartype.typing import List, Literal, Optional, Sequence, Tuple
from rasterio import warp
from rasterio.enums import Resampling
from rasterio.profiles import Profile

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.raster_processing.resampling import RESAMPLE_METHOD_MAP


def _calculate_snapped_grid(
    raster: rasterio.io.DatasetReader, dst_crs: rasterio.crs.CRS, dst_resolution: Tuple[float, float]
) -> Tuple[warp.Affine, int, int]:
    dst_transform, dst_width, dst_height = warp.calculate_default_transform(
        raster.crs, dst_crs, raster.width, raster.height, *raster.bounds, resolution=dst_resolution
    )
    # The created transform might not be aligned with the base raster grid, so
    # we still need to snap/align the transformation to closest grid corner
    x_distance_to_grid = dst_transform.c % dst_resolution[0]
    y_distance_to_grid = dst_transform.f % dst_resolution[1]

    if x_distance_to_grid > dst_resolution[0] / 2:  # Snap towards right
        c = dst_transform.c - x_distance_to_grid + dst_resolution[0]
    else:  # Snap towards left
        c = dst_transform.c - x_distance_to_grid

    if y_distance_to_grid > dst_resolution[1] / 2:  # Snap towards up
        f = dst_transform.f - y_distance_to_grid + dst_resolution[1]
    else:  # Snap towards bottom
        f = dst_transform.f - y_distance_to_grid

    # Create new transform with updated corner coordinates
    dst_transform = warp.Affine(
        dst_transform.a,  # Pixel size x
        dst_transform.b,  # Shear parameter
        c,  # Up-left corner x-coordinate
        dst_transform.d,  # Shear parameter
        dst_transform.e,  # Pixel size y
        f,  # Up-left corner y-coordinate
    )
    return dst_transform, dst_width, dst_height


def _mask_nodata(
    raster_array: np.ndarray,
    nodata_value: Optional[float],
    base_raster_array: np.ndarray,
    base_raster_profile: rasterio.profiles.Profile,
):
    # Take only first band as nodata mask from base raster
    if base_raster_array.ndim != 1:
        base_raster_array = base_raster_array[0]

    # Create the mask
    base_nodata_value = base_raster_profile.get("nodata", np.nan)
    mask = (base_raster_array == base_nodata_value) | np.isnan(base_raster_array)

    # Apply to each band
    if raster_array.ndim != 1:
        for array in raster_array:
            array[mask] = nodata_value
    else:
        raster_array[mask] = nodata_value


def _unify_raster_grids(
    base_raster: rasterio.io.DatasetReader,
    rasters_to_unify: Sequence[rasterio.io.DatasetReader],
    resampling_method: Resampling,
    masking: Optional[Literal["extents", "full"]],
) -> List[Tuple[np.ndarray, Profile]]:

    dst_crs = base_raster.crs
    dst_width = base_raster.width
    dst_height = base_raster.height
    dst_transform = base_raster.transform
    dst_resolution = (base_raster.transform.a, abs(base_raster.transform.e))

    base_raster_arr = base_raster.read()
    base_raster_profile = base_raster.profile.copy()
    out_rasters = [(base_raster_arr, base_raster_profile)]

    for raster in rasters_to_unify:
        out_profile = raster.profile.copy()

        # If we unify without clipping, things are more complicated and we need to
        # calculate corner coordinates, width and height, and snap the grid to nearest corner
        if not masking:
            dst_transform, dst_width, dst_height = _calculate_snapped_grid(raster, dst_crs, dst_resolution)

        dst_array = np.empty((base_raster.count, dst_height, dst_width))
        base_raster_nodata = base_raster_profile.get("nodata", np.nan)
        nodata = out_profile["nodata"]
        if nodata is None:
            nodata = base_raster_nodata
            out_profile["nodata"] = base_raster_nodata

        dst_array.fill(nodata)

        src_array = raster.read()

        out_image = warp.reproject(
            source=src_array,
            src_crs=raster.crs,
            src_transform=raster.transform,
            src_nodata=nodata,
            destination=dst_array,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            dst_nodata=nodata,
            resampling=resampling_method,
        )[0]

        if masking == "full":
            _mask_nodata(out_image, nodata, base_raster_arr, base_raster_profile)

        out_profile.update({"transform": dst_transform, "width": dst_width, "height": dst_height, "crs": dst_crs})

        out_rasters.append((out_image, out_profile))

    return out_rasters


@beartype
def unify_raster_grids(
    base_raster: rasterio.io.DatasetReader,
    rasters_to_unify: Sequence[rasterio.io.DatasetReader],
    resampling_method: Literal["nearest", "bilinear", "cubic", "average", "gauss", "max", "min"] = "nearest",
    masking: Optional[Literal["extents", "full"]] = "extents",
) -> List[Tuple[np.ndarray, Profile]]:
    """Unifies given rasters with the base raster.

    Performs the following operations:
    - Reprojecting
    - Resampling
    - Aligning / snapping
    - Clipping / expanding extents (optional)
    - Copying nodata cells from base raster (optional)

    Args:
        base_raster: The base raster to determine target raster grid properties.
        rasters_to_unify: Rasters to be unified with the base raster.
        resampling_method: Resampling method. Most suitable method depends on the dataset and context.
            `nearest`, `bilinear` and `cubic` are some common choices. This parameter defaults to `nearest`.
        masking: Controls if and how masking should be handled. If `extents`, the bounds of rasters to-be-unified
            are matched with the base raster. Larger rasters are clipped and smaller rasters expanded (with nodata).
            If `full`, copies nodata pixel locations from the base raster additionally. If None,
            extents are not matched and nodata not copied. Defaults to `extents`.

    Returns:
        List of unified rasters' data and profiles. First element is the base raster.

    Raises:
        InvalidParameterValueException: Rasters to unify is empty.
    """
    if len(rasters_to_unify) == 0:
        raise InvalidParameterValueException("Rasters to unify is empty.")

    method = RESAMPLE_METHOD_MAP[resampling_method]
    out_rasters = _unify_raster_grids(base_raster, rasters_to_unify, method, masking)
    return out_rasters
