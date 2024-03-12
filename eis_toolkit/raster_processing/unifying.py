import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import List, Literal, Sequence, Tuple
from rasterio import warp
from rasterio.enums import Resampling

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.raster_processing.resampling import RESAMPLE_METHOD_MAP


def _unify_raster_grids(
    base_raster: rasterio.io.DatasetReader,
    rasters_to_unify: Sequence[rasterio.io.DatasetReader],
    resampling_method: Resampling,
    same_extent: bool,
) -> List[Tuple[np.ndarray, dict]]:

    dst_crs = base_raster.crs
    dst_width = base_raster.width
    dst_height = base_raster.height
    dst_transform = base_raster.transform
    dst_resolution = (base_raster.transform.a, abs(base_raster.transform.e))

    out_rasters = [(base_raster.read(), base_raster.meta.copy())]
    out_meta = base_raster.meta.copy()

    for raster in rasters_to_unify:

        # If we unify without clipping, things are more complicated and we need to
        # calculate corner coordinates, width and height, and snap the grid to nearest corner
        if not same_extent:
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

            out_meta["transform"] = dst_transform
            out_meta["width"] = dst_width
            out_meta["height"] = dst_height

        # Initialize output raster arrary
        dst_array = np.empty((base_raster.count, dst_height, dst_width))
        dst_array.fill(base_raster.meta["nodata"])

        src_array = raster.read()

        out_image = warp.reproject(
            source=src_array,
            src_crs=raster.crs,
            src_transform=raster.transform,
            src_nodata=raster.meta["nodata"],
            destination=dst_array,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            dst_nodata=base_raster.meta["nodata"],
            resampling=resampling_method,
        )[0]

        out_rasters.append((out_image, out_meta))

    return out_rasters


@beartype
def unify_raster_grids(
    base_raster: rasterio.io.DatasetReader,
    rasters_to_unify: Sequence[rasterio.io.DatasetReader],
    resampling_method: Literal["nearest", "bilinear", "cubic", "average", "gauss", "max", "min"] = "nearest",
    same_extent: bool = False,
) -> List[Tuple[np.ndarray, dict]]:
    """Unifies (reprojects, resamples, aligns and optionally clips) given rasters relative to base raster.

    Args:
        base_raster: The base raster to determine target raster grid properties.
        rasters_to_unify: Rasters to be unified with the base raster.
        resampling_method: Resampling method. Most suitable
            method depends on the dataset and context. Nearest, bilinear and cubic are some
            common choices. This parameter defaults to nearest.
        same_extent: If the unified rasters will be forced to have the same extent/bounds
            as the base raster. Expands smaller rasters with nodata cells. Defaults to False.

    Returns:
        List of unified rasters' data and metadata. First element is the base raster.

    Raises:
        InvalidParameterValueException: Rasters to unify is empty.
    """
    if len(rasters_to_unify) == 0:
        raise InvalidParameterValueException("Rasters to unify is empty.")

    method = RESAMPLE_METHOD_MAP[resampling_method]
    out_rasters = _unify_raster_grids(base_raster, rasters_to_unify, method, same_extent)
    return out_rasters
