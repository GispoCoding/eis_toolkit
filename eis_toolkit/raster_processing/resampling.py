from numbers import Number
from typing import Optional

import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Tuple
from rasterio.enums import Resampling

from eis_toolkit.checks.parameter import check_numeric_value_sign
from eis_toolkit.exceptions import NumericValueSignException


# The core resampling functionality. Used internally by resample.
def _resample(
    raster: rasterio.io.DatasetReader,
    resampling_method: Resampling,
    upscale_factor: Number,
    upscale_factor_y: Optional[Number],
) -> Tuple[np.ndarray, dict]:

    if upscale_factor_y is None:
        upscale_factor_y = upscale_factor

    out_image = raster.read(
        out_shape=(raster.count, round(raster.height * upscale_factor), round(raster.width * upscale_factor_y)),
        resampling=resampling_method,
    )

    out_transform = raster.transform * raster.transform.scale(
        (raster.width / out_image.shape[-1]), (raster.height / out_image.shape[-2])
    )

    out_meta = raster.meta.copy()
    out_meta.update(
        {
            "transform": out_transform,
            "width": out_image.shape[-1],
            "height": out_image.shape[-2],
        }
    )

    return out_image, out_meta


@beartype
def resample(
    raster: rasterio.io.DatasetReader,
    upscale_factor: Number,
    upscale_factor_y: Optional[Number] = None,
    resampling_method: Resampling = Resampling.bilinear,
) -> Tuple[np.ndarray, dict]:
    """Resamples raster according to given upscale factor.

    Args:
        raster: The raster to be resampled.
        upscale_factor: Resampling factor for raster width (and height by default).
            Scale factors over 1 will yield higher resolution data. Value must be positive.
        upscale_factor_y: Resampling factor for raster height, if different scaling is needed
            for x and y directions. Defaults to None, in which case upscale_factor is used
            for both width and height.
        resampling_method: Resampling method. Most suitable
            method depends on the dataset and context. Nearest, bilinear and cubic are some
            common choices. This parameter defaults to bilinear.

    Returns:
        The resampled raster data.
        The updated metadata.

    Raises:
        NumericValueSignException: Upscale factor (y) is not a positive value.
    """
    if not check_numeric_value_sign(upscale_factor):
        raise NumericValueSignException(f"Upscale factor is not a positive value: {upscale_factor}")
    if upscale_factor_y is not None and not check_numeric_value_sign(upscale_factor_y):
        raise NumericValueSignException(f"Upscale factor y is not a positive value: {upscale_factor_y}")

    out_image, out_meta = _resample(raster, resampling_method, upscale_factor, upscale_factor_y)
    return out_image, out_meta
