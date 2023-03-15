import rasterio
import numpy as np

from rasterio.enums import Resampling
from rasterio import warp
from typing import List, Tuple

from eis_toolkit.exceptions import InvalidParameterValueException


def _coregister_rasters(
    base_raster: rasterio.io.DatasetReader,
    rasters_to_unify: List[rasterio.io.DatasetReader],
    resampling_method: Resampling,
    same_extent: bool
)  -> List[Tuple[np.ndarray, dict]]:

    # Open the base raster with target CRS, resolution and alignment
    dst_crs = base_raster.crs
    dst_width = base_raster.width
    dst_height = base_raster.height
    dst_transform = base_raster.transform
    dst_resolution = (base_raster.transform.a, abs(base_raster.transform.e))
    
    out_rasters = [(base_raster.read(), base_raster.meta.copy())]
    out_meta = base_raster.meta.copy()

    for raster in rasters_to_unify:

        # If we don't want to clip the rasters, things are more complicated...
        if not same_extent:
            dst_transform, dst_width, dst_height = warp.calculate_default_transform(
                raster.crs, dst_crs, raster.width, raster.height, *raster.bounds,
                resolution=dst_resolution
            )
            # The created transform might not be aligned with the base raster grid, so
            # we still need to align the transformation to closest grid corner
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

            # Create new transform
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

        # Initialize output raster arr
        dst_arr = np.empty((base_raster.count, dst_height, dst_width))
        dst_arr.fill(base_raster.meta["nodata"])

        src_arr = raster.read()

        out_image = warp.reproject(
            source=src_arr,
            src_crs=raster.crs,
            src_transform=raster.transform,
            src_nodata=base_raster.meta["nodata"],
            destination=dst_arr,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            dst_nodata=base_raster.meta["nodata"],
            resampling=resampling_method
        )[0]

        out_rasters.append((out_image, out_meta))

    return out_rasters


def coregister_rasters(  # type: ignore[no-any-unimported]
    base_raster: rasterio.io.DatasetReader,
    raster_list: List[rasterio.io.DatasetReader],
    resampling_method: Resampling = Resampling.bilinear,
    same_extent: bool = True
) -> List[Tuple[np.ndarray, dict]]:
    """Coregisters/unifies (reprojects, resamples and aligns) given rasters relative to a base raster.

    Args:
        raster (rasterio.io.DatasetReader): The base raster to determine unifying.
        raster_list (list(rasterio.io.DatasetReader)): List of rasters to be unified.
        resampling_method (rasterio.enums.Resampling): Resampling method. Most suitable
            method depends on the dataset and context. Nearest, bilinear and cubic are some
            common choices. This parameter defaults to bilinear.
        same_extent (bool): If the processed rasters will be forced to have the same extent/bounds
            as the base raster. Defaults to True.

    Returns:
        out_rasters (list(tuple(numpy.ndarray, dict))): List of unified rasters' data and metadata.
            First element is the base raster.

    Raises:
        InvalidParameterValueException
    """
    if not isinstance(base_raster, rasterio.io.DatasetReader):
        raise InvalidParameterValueException
    if not isinstance(raster_list, list):
        raise InvalidParameterValueException
    if not all(isinstance(raster, rasterio.io.DatasetReader) for raster in raster_list):
        raise InvalidParameterValueException
    if len(raster_list) == 0:
        raise InvalidParameterValueException

    out_rasters = _coregister_rasters(base_raster, raster_list, resampling_method, same_extent)
    return out_rasters
