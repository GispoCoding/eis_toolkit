import rasterio
import numpy as np

from numbers import Number
from beartype import beartype
from beartype.typing import Optional, Union, Literal

from eis_toolkit.surface_attributes.partial_derivatives import _method_horn
from eis_toolkit.utilities.nodata import nodata_to_nan, nan_to_nodata
from eis_toolkit.utilities.conversions import convert_rad_to_degree
from eis_toolkit.checks.raster import check_quadratic_pixels
from eis_toolkit.exceptions import InvalidRasterBandException, NonSquarePixelSizeException


@beartype
def _get_slope(
    raster: rasterio.io.DatasetReader,
    method: Literal,
    scaling_factor: Number,
) -> tuple(np.ndarray, dict):

    cellsize = raster.res[0]
    out_meta = raster.meta.copy()
    out_array = np.squeeze(raster.read())

    out_array = nodata_to_nan(out_array, nodata_value=raster.nodata)
    out_array = out_array * scaling_factor

    if method == "Horn81" and any(coefficient is None for coefficient in (p, q)):
        p = _method_horn(out_array, cellsize=cellsize, parameter="p")
        q = _method_horn(out_array, cellsize=cellsize, parameter="q")

    out_array = np.sqrt(np.square(p) + np.square(q))
    out_array = np.arctan(out_array) * (180.0 / np.pi)

    out_array = nan_to_nodata(out_array, nodata_value=raster.nodata).astype(np.float16)
    out_meta.update({"dtype": out_array.dtype.name})

    return out_array, out_meta


@beartype
def get_slope(
    raster: rasterio.io.DatasetReader,
    method: Literal = "Horn81",
    scaling_factor: Optional[Number] = 1,
) -> tuple(np.ndarray, dict):
    """
    Calculate the slope of a given surface.

    Args:
        raster: The input raster data.
        method: Basic method used to calculate partial derivatives.
        scaling_factor: Factor to modify values, e.g. for unit conversion.

    Returns:
        The calculated slope in degree (0-90).
    """

    if raster.count > 1:
        raise InvalidRasterBandException("Only one-band raster supported.")

    if check_quadratic_pixels(raster) is False:
        raise NonSquarePixelSizeException("Processing requires quadratic pixel dimensions.")

    return _get_slope(raster, method, scaling_factor)
