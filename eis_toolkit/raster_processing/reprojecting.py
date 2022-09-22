from typing import Tuple

import numpy as np
import rasterio
from rasterio import warp

from eis_toolkit.exceptions import MatchingCrsException


# Core reprojecting functionality used internally by reproject_raster and reproject_and_write_raster
def _reproject_raster(  # type: ignore[no-any-unimported]
    src: rasterio.io.DatasetReader, target_EPSG: int, resampling_method: warp.Resampling
) -> Tuple[np.ndarray, dict]:

    src_arr = src.read()
    dst_crs = rasterio.CRS.from_epsg(target_EPSG)

    dst_transform, dst_width, dst_height = warp.calculate_default_transform(
        src.crs,
        dst_crs,
        src.width,
        src.height,
        *src.bounds,
    )

    # Initialize base raster (target raster)
    dst = np.empty((src.count, dst_height, dst_width))
    dst.fill(-9999)

    out_image = warp.reproject(
        source=src_arr,
        src_transform=src.transform,
        src_crs=src.crs,
        destination=dst,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=-9999,
        dst_nodata=src.meta["nodata"],
        resampling=resampling_method,
    )[0]

    out_meta = src.meta.copy()
    out_meta.update(
        {
            "crs": dst_crs,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
        }
    )

    return out_image, out_meta


def reproject_raster(  # type: ignore[no-any-unimported]
    src: rasterio.io.DatasetReader, target_EPSG: int, resampling_method: int = 0
) -> Tuple[np.ndarray, dict]:
    """Reprojects raster to match given coordinate system (EPSG).

    Args:
        raster (rasterio.io.DatasetReader): The raster to be clipped.
        target_EPSG (int): Target crs as EPSG code.
        resampling_method (int): Resampling method. Can be either 0, 1 or 2 that correspond to 'nearest', 'bilinear' or
        'cubic' respectively.

    Returns:
        out_image (numpy.ndarray): Reprojected raster data.
        out_meta (dict): The updated metadata.
    """
    if target_EPSG == int(src.crs.to_string()[5:]):
        raise MatchingCrsException

    resamplers = {
        0: warp.Resampling.nearest,
        1: warp.Resampling.bilinear,
        2: warp.Resampling.cubic,
    }
    resampler = resamplers[resampling_method]

    out_image, out_meta = _reproject_raster(src, target_EPSG, resampler)

    return out_image, out_meta
