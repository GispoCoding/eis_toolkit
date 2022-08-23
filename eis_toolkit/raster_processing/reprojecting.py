import rasterio
import numpy as np
from rasterio import warp
from typing import Tuple

def reproject(
    raster: rasterio.io.DatasetReader,
    target_EPSG: int,
    resampling_method: str = "nearest"
) -> Tuple[np.ndarray, dict]:
    """Reprojects raster to match given coordinate system (EPSG).

    Args:
        raster (rasterio.io.DatasetReader): The raster to be clipped.
        target_EPSG (int): Target crs as EPSG code.
        resampling_method (str): Resampling method. Can be either 'nearest', 'bilinear'
            or 'cubic'

    Returns:
        out_image (numpy.ndarray): Reprojected raster data.
        out_meta (dict): The updated metadata.
    """

    resamplers =  {
        'nearest': warp.Resampling.nearest,
        'bilinear': warp.Resampling.bilinear,
        'cubic': warp.Resampling.cubic
    }

    # Compare input and target crs and do nothing if they match
    if target_EPSG == int(raster.crs.to_string()[5:]):
        out_image = raster.read()
        out_meta = raster.meta.copy()
    
    # Else reproject the input raster
    else:
        raster_arr = raster.read()
        br_crs = rasterio.CRS.from_epsg(target_EPSG)

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

        # Initialize base raster (target raster)
        br = np.zeros((raster.count, br_height, br_width))
        
        out_image = warp.reproject(
            source=raster_arr,
            src_transform=raster.transform,
            src_crs=raster.crs,
            destination=br,
            dst_transform=br_transform,
            dst_crs=br_crs,
            resampling=resamplers[resampling_method]
        )[0]

        out_meta = raster.meta.copy()
        out_meta.update({
            'crs': br_crs,
            'transform': br_transform,
            "width": br_width,
            "height": br_height,
        })

    return out_image, out_meta