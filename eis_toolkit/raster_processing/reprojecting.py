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
    dst_crs = rasterio.crs.CRS.from_epsg(target_EPSG)

    dst_transform, dst_width, dst_height = warp.calculate_default_transform(
        src.crs,
        dst_crs,
        src.width,
        src.height,
        *src.bounds,
    )

    # Initialize base raster (target raster)
    dst = np.empty((src.count, dst_height, dst_width))
    dst.fill(src.meta["nodata"])

    out_image = warp.reproject(
        source=src_arr,
        src_transform=src.transform,
        src_crs=src.crs,
        destination=dst,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=src.meta["nodata"],
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
    src: rasterio.io.DatasetReader, target_EPSG: int, resampling_method: warp.Resampling = warp.Resampling.nearest
) -> Tuple[np.ndarray, dict]:
    """Reprojects raster to match given coordinate system (EPSG).

    Args:
        raster (rasterio.io.DatasetReader): The raster to be clipped.
        target_EPSG (int): Target crs as EPSG code.
        resampling_method (warp.Resampling): Resampling method. Most suitable
            method depends on the dataset and context. Nearest, bilinear and cubic are some
            common choices. This parameter defaults to nearest.

    Returns:
        out_image (numpy.ndarray): Reprojected raster data.
        out_meta (dict): The updated metadata.
    """
    if target_EPSG == int(src.crs.to_string()[5:]):
        raise MatchingCrsException

    out_image, out_meta = _reproject_raster(src, target_EPSG, resampling_method)

    return out_image, out_meta
