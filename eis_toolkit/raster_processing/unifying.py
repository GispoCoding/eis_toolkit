from typing import Tuple, List

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.raster_processing import reprojecting, resampling, snapping


# The core unifying functionality. Used internally by unify_rasters.
def _unify_rasters(  # type: ignore[no-any-unimported]
    base_raster: rasterio.io.DatasetReader,
    raster_list: List[rasterio.io.DatasetReader],
    resampling_method: Resampling
) -> List[Tuple[np.ndarray, dict]]:

    target_epsg = int(base_raster.crs.to_string()[5:])
    # bbox = base_raster.bounds
    # target_center_coords = ((bbox.left + bbox.right) / 2, (bbox.top + bbox.bottom) / 2)
    # target_width = base_raster.width
    # target_height = base_raster.height

    # TODO: now it is expected all subfunctions go through without problems
    base_image = base_raster.read()
    base_meta = base_raster.meta.copy()
    out_rasters = [(base_image, base_meta)]

    for raster in raster_list:
        # Reproject
        if raster.crs != base_raster.crs:
            out_image, out_meta = reprojecting.reproject_raster(raster, target_epsg, resampling_method)
        else:
            out_image, out_meta = raster.read(), raster.meta.copy()

        # Save to memory, then resample
        upscale_factor_x = base_raster.transform.a / out_meta["transform"].a 
        # upscale_factor_y = base_raster.transform.e / out_meta.transform.e
        # TODO: modify resample function to accept x and y scale factors differently
        if upscale_factor_x != 1:
            with MemoryFile() as memfile:
                with memfile.open('w', driver='GTiff', **out_meta) as dataset:
                    dataset.write(out_image)
                with memfile.open() as reprojected_raster:
                    out_image, out_meta = resampling.resample(
                        reprojected_raster, upscale_factor_x, resampling_method
                    )

        # Save to memory, then snap
        with MemoryFile() as memfile:
            with memfile.open('w', driver='GTiff', **out_meta) as dataset:
                dataset.write(out_image)
            with memfile.open() as resampled_raster:
                out_image, out_meta = snapping.snap_with_raster(
                    resampled_raster, base_raster
                )

        out_rasters.append((out_image, out_meta))

    return out_rasters


def unify_rasters(  # type: ignore[no-any-unimported]
    base_raster: rasterio.io.DatasetReader,
    raster_list: List[rasterio.io.DatasetReader],
    resampling_method: Resampling = Resampling.bilinear
) -> List[Tuple[np.ndarray, dict]]:
    """Unifies (reprojects, resamples and aligns) given rasters relative to a base raster.

    Args:
        raster (rasterio.io.DatasetReader): The base raster to determine unifying.
        raster_list (list(rasterio.io.DatasetReader)): List of rasters to be unified.
        resampling_method (rasterio.enums.Resampling): Resampling method. Most suitable
            method depends on the dataset and context. Nearest, bilinear and cubic are some
            common choices. This parameter defaults to bilinear.

    Returns:
        out_rasters (list(tuple(numpy.ndarray, dict))): List of unified rasters' data and metadata.
            First element is the base raster.

    Raises:
        InvalidParameterValueException: Upscale factor is not a positive value.
    """
    if not isinstance(base_raster, rasterio.io.DatasetReader):
        raise InvalidParameterValueException
    if not isinstance(raster_list, list):
        raise InvalidParameterValueException
    if not all(isinstance(raster, rasterio.io.DatasetReader) for raster in raster_list):
        raise InvalidParameterValueException

    out_rasters = _unify_rasters(base_raster, raster_list, resampling_method)
    return out_rasters
