import rasterio
import numpy as np
from rasterio.enums import Resampling
from typing import Tuple

from eis_toolkit.checks.parameter import check_parameter_value, check_resample_upscale_factor
from eis_toolkit.exceptions import NegativeResamplingFactorException, InvalidParameterValueException


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


def resample(raster: rasterio.io.DatasetReader,
    upscale_factor: float,
    resampling_method: int = 1
) -> Tuple[np.ndarray, dict]:
    """Resamples raster according to given upscale factor.

    Args:
        raster (rasterio.io.DatasetReader): The raster to be resampled.
        upscale_factor (float): Resampling factor. Scale factors over 1 will yield
            higher resolution data. Value must be positive.
        resampling_method (int): Parameterized resampling method. Options are
            0: nearest,
            1: bilinear,
            2: cubic.
            Defaults to bilinear.

    Returns:
        out_image (numpy.ndarray): Resampled raster data.
        out_meta (dict): The updated metadata.

    Raises:
        NegativeResamplingFactorException: Upscale factor is negative (or not positive).
        InvalidParameterValue: Resample method parameter did not correspond to any method.
    """
    if not check_parameter_value(
        parameter_value = resampling_method,
        allowed_values = [0, 1, 2]
    ):
        raise InvalidParameterValueException

    if not check_resample_upscale_factor(
        upscale_factor
    ):
        raise NegativeResamplingFactorException

    resamplers =  {
        0: Resampling.nearest,
        1: Resampling.bilinear,
        2: Resampling.cubic
    }

    out_image, out_meta = _resample(raster, upscale_factor, resamplers[resampling_method])
    return out_image, out_meta