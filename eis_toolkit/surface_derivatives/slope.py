import scipy
import rasterio
import numpy as np

from beartype import beartype
from beartype import Literal
from numbers import Number

from eis_toolkit.utilities import conversions
from eis_toolkit.utilities.nodata import nodata_to_nan, nan_to_nodata


@beartype
def _get_slope(p, q, unit: Literal) -> np.ndarray:

    out_array = np.sqrt(np.square(p) + np.square(q))
    out_array = np.arctan(out_array)

    if unit == "radians":
        return out_array
    elif unit == "degree":
        return conversions.convert_rad_to_degree(out_array)
    elif unit == "rise":
        return conversions.convert_rad_to_rise(out_array)


def get_slope(raster: rasterio.io.DatasetReader, unit: Literal = "radians") -> np.ndarray:
    """
    Calculate the slope of a given surface.

    Unit options are: 'radians', 'degree' and 'rise'.

    Args:
        raster: The input raster data.
        unit: The unit of the calculated slope.

    Returns:
        The calculated slope.
    """

    # exceptions
    ## bands: maybe not, can also work with all bands ...
    ## quadratic pixel

    out_array = raster.read()
    out_array = nodata_to_nan(out_array, nodata_value=raster.nodata)

    # partial derivatives...

    out_array = _get_slope(out_array, unit=unit)
    out_array = nan_to_nodata(out_array, nodata_value=raster.nodata)

    return out_array
