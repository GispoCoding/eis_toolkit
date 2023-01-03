from typing import Tuple, List

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.mask import mask

from eis_toolkit.checks.parameter import check_numeric_value_sign
from eis_toolkit.exceptions import NumericValueSignException

from eis_toolkit.raster_processing import reprojecting, resampling, snapping, clipping


# The core resampling functionality. Used internally by resample.
def _unify_rasters(  # type: ignore[no-any-unimported]
    base_raster: rasterio.io.DatasetReader,
    rasters: List[rasterio.io.DatasetReader],
    resampling_method: Resampling
) -> Tuple[np.ndarray, dict]:

    # Reproject, resample, snap, clip

    target_epsg = int(base_raster.crs.to_string()[5:])
    target_pixel_size_x = base_raster.transform.a
    target_pixel_size_y = base_raster.transform.e

    # TODO: now it is expected all subfunctions go through without problems
    for raster in rasters:
        # Reproject
        out_image, out_meta = reprojecting.reproject_raster(raster, target_epsg, resampling_method)

        # Save to memory, then resample
        raster_pixel_size_x = out_meta.transform.a
        raster_pixels_size_y = out_meta.transform.e
        upscale_factor_x = target_pixel_size_x / raster_pixel_size_x
        upscale_factor_y = target_pixel_size_y / raster_pixels_size_y
        # TODO: modify resample function to accept x and y scale factors differently
        with MemoryFile() as memfile:
            with memfile.open('w', driver='GTiff', **out_meta) as dataset:
                dataset.write(out_image)
            with memfile.open() as reprojected_raster:
                out_image, out_meta = resampling.resample(reprojected_raster, upscale_factor_x, resampling_method)
            
        # Save to memory, then snap
        with MemoryFile() as memfile:
            with memfile.open('w', driver='GTiff', **out_meta) as dataset:
                dataset.write(out_image)
            with memfile.open() as resampled_raster:
                out_image, out_meta = snapping.snap_with_raster(resampled_raster, base_raster, resampled_raster.count)
        
        # Save to memory, then clip
        with MemoryFile() as memfile:
            with memfile.open('w', driver='GTiff', **out_meta) as dataset:
                dataset.write(out_image)
            with memfile.open() as snapped_raster:
                pass
                # TODO: Decide which to use: clipping, rasterio.mask, or something else?

    return out_image, out_meta

with rasterio.open('example.tif', 'w', **meta) as dst:
    dst.write(array.astype(rasterio.uint8), 1)

def unify_rasters(  # type: ignore[no-any-unimported]
    base_raster: rasterio.io.DatasetReader,
    rasters: List[rasterio.io.DatasetReader],
    resampling_method: Resampling = Resampling.bilinear
) -> Tuple[np.ndarray, dict]:
    """Resamples raster according to given upscale factor.

    Args:
        raster (rasterio.io.DatasetReader): The raster to be resampled.
        upscale_factor (float): Resampling factor. Scale factors over 1 will yield
            higher resolution data. Value must be positive.
        resampling_method (rasterio.enums.Resampling): Resampling method. Most suitable
            method depends on the dataset and context. Nearest, bilinear and cubic are some
            common choices. This parameter defaults to bilinear.

    Returns:
        out_image (numpy.ndarray): Resampled raster data.
        out_meta (dict): The updated metadata.

    Raises:
        NumericValueSignException: Upscale factor is not a positive value.
    """
    if not check_numeric_value_sign(upscale_factor):
        raise NumericValueSignException

    out_image, out_meta = _resample(raster, upscale_factor, resampling_method)
    return out_image, out_meta
