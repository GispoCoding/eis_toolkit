import rasterio
import numpy as np
from rasterio.enums import Resampling
from typing import Tuple


def resample_raster(
    raster: rasterio.io.DatasetReader,
    upscale_factor: float,
    resampling_method: str = "bilinear"
) -> Tuple[np.ndarray, dict]:
    """Resamples raster according to given upscale factor.

    Args:
        raster (rasterio.io.DatasetReader): The raster to be resampled.
        upscale_factor (float): Resampling factor. Scale factors over 1 will yield
            higher resolution data.
        resampling_method (str): Resampling method. Can be either 'nearest', 'bilinear'
            or 'cubic'. Defaults to bilinear.

    Returns:
        out_image (numpy.ndarray): Resampled raster data.
        out_meta (dict): The updated metadata.
    """

    resamplers =  {
        'nearest': Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'cubic': Resampling.cubic
    }

    out_image = raster.read(
        out_shape=(
            raster.count,
            int(raster.height * upscale_factor),
            int(raster.width * upscale_factor)
        ),
        resampling=resamplers[resampling_method]
    )

    out_transform = raster.transform * raster.transform.scale(
        (raster.width / out_image.shape[-1]),
        (raster.height / out_image.shape[-2])
    )

    out_meta = raster.meta.copy()
    out_meta.update({
            'transform': out_transform,
            'width': out_image.shape[-1],
            'height': out_image.shape[-2],
            'nodata': 0,
        })

    return out_image, out_meta