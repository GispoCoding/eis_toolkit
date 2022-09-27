import rasterio
import numpy as np
from rasterio.enums import Resampling
from typing import Tuple

from eis_toolkit.checks.parameter import check_numeric_value_sign
from eis_toolkit.exceptions import NumericValueSignException


def _resample(
    raster: rasterio.io.DatasetReader, upscale_factor: float, resampling_method: Resampling
) -> Tuple[np.ndarray, dict]:

    out_image = raster.read(
        out_shape=(
            raster.count,
            int(raster.height * upscale_factor),
            int(raster.width * upscale_factor)
        ),
        resampling=resampling_method
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


def resample(raster: rasterio.io.DatasetReader, upscale_factor: float, resampling_method: Resampling
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
    if not check_numeric_value_sign(
        upscale_factor
    ):
        raise NumericValueSignException

    out_image, out_meta = _resample(raster, upscale_factor, resampling_method)
    return out_image, out_meta