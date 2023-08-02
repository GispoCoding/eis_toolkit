from numbers import Number

import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Literal, Optional

from eis_toolkit.checks.raster import check_quadratic_pixels
from eis_toolkit.exceptions import InvalidRasterBandException, NonSquarePixelSizeException
from eis_toolkit.surface_attributes.partial_derivatives import _method_horn
from eis_toolkit.utilities.conversions import _convert_rad_to_rise, convert_rad_to_degree
from eis_toolkit.utilities.nodata import nan_to_nodata, nodata_to_nan


@beartype
def _get_slope(
    raster: rasterio.io.DatasetReader,
    method: Literal["Horn81"],
    scaling_factor: Number,
    unit: Literal["degree", "rise"],
) -> tuple[np.ndarray, dict]:

    cellsize = raster.res[0]
    out_meta = raster.meta.copy()
    nodata = -9999

    out_array = raster.read()
    if out_array.ndim >= 3:
        out_array = np.squeeze(out_array)

    out_array = nodata_to_nan(out_array, nodata_value=raster.nodata)
    out_array = out_array * scaling_factor

    if method == "Horn81":
        p = _method_horn(out_array, cellsize=cellsize, parameter="p")
        q = _method_horn(out_array, cellsize=cellsize, parameter="q")

    out_array = np.sqrt(np.square(p) + np.square(q))
    out_array = np.arctan(out_array)

    if unit == "degree":
        out_array = convert_rad_to_degree(out_array)
    elif unit == "rise":
        out_array = _convert_rad_to_rise(out_array)

    out_array = nan_to_nodata(out_array, nodata_value=nodata).astype(np.float32)
    out_meta.update({"dtype": out_array.dtype.name, "nodata": nodata})

    return out_array, out_meta


@beartype
def get_slope(
    raster: rasterio.io.DatasetReader,
    method: Literal["Horn81"] = "Horn81",
    unit: Literal["degree", "rise"] = "degree",
    scaling_factor: Optional[Number] = 1,
) -> tuple[np.ndarray, dict]:
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

    return _get_slope(raster, method, scaling_factor, unit)
