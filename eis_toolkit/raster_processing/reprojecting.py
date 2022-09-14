import rasterio
import numpy as np
from rasterio import warp
from typing import Tuple

def reproject_raster(
    src: rasterio.io.DatasetReader,
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
    if target_EPSG == int(src.crs.to_string()[5:]):
        out_image = src.read()
        out_meta = src.meta.copy()
    
    # Else reproject the input raster
    else:
        src_arr = src.read()
        dst_crs = rasterio.CRS.from_epsg(target_EPSG)

        dst_transform, dst_width, dst_height = warp.calculate_default_transform(
           src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        # Initialize base raster (target raster)
        dst = np.zeros((src.count, dst_height, dst_width))
        
        out_image = warp.reproject(
            source=src_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            destination=dst,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resamplers[resampling_method]
        )[0]

        out_meta = src.meta.copy()
        out_meta.update({
            'crs': dst_crs,
            'transform': dst_transform,
            "width": dst_width,
            "height": dst_height,
        })

    return out_image, out_meta