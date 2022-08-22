from curses.panel import bottom_panel
import rasterio
from rasterio import warp
import numpy as np

from typing import Tuple, Optional
from eis_toolkit.exceptions import InvalidRasterDimension

def reproject(
    raster: rasterio.io.DatasetReader,
    br_EPSG: int,
) -> Tuple[np.ndarray, dict]:
    """Reprojects raster to match base raster (br).

    Args:
        raster (rasterio.io.DatasetReader): The raster to be clipped.
        dst_EPSG (int): Base raster crs as EPSG code.

    Returns:
        out_image (numpy.ndarray): Reprojected raster data
        out_meta (dict): The updated metadata

    Raises:
        ?
    """

    # Compare crs and do nothing if they match

    br_crs = rasterio.CRS.from_epsg(br_EPSG)

    br_transform, br_width, br_height = warp.calculate_default_transform(
    src_crs=raster.crs,
    dst_crs=br_crs,
    width=raster.width,
    height=raster.height,
    left=raster.bounds.left,
    right=raster.bounds.right,
    top=raster.bounds.top,
    bottom=raster.bounds.bottom
    )

    br = np.zeros((raster.count, br_height, br_width))
    raster_arr = raster.read()
    
    out_image = rasterio.warp.reproject(
        source=raster_arr,
        src_transform=raster.transform,
        src_crs=raster.crs,
        destination=br,
        dst_transform=br_transform,
        dst_crs=br_crs
    )

    out_meta = raster.meta.copy()
    out_meta.update({
        'crs': br_crs,
        'transform': br_transform
    })

    return out_image, out_meta
