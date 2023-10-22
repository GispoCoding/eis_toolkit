import numpy as np
import rasterio
from numbers import Number
from beartype import beartype
from beartype.typing import Literal, Optional, Sequence

from eis_toolkit.checks.raster import check_quadratic_pixels
from eis_toolkit.exceptions import (
    InvalidRasterBandException,
    NonSquarePixelSizeException,
    InvalidParameterValueException,
    NumericValueSignException,
)
from eis_toolkit.surface_attributes.partial_derivatives import coefficients
from eis_toolkit.surface_attributes.gradient import slope, aspect
from eis_toolkit.surface_attributes.utilities import set_flat_pixels, scale_raster, reduce_ndim
from eis_toolkit.utilities.conversions import convert_rad_to_rise, convert_rad_to_deg
from eis_toolkit.utilities.nodata import nan_to_nodata, nodata_to_nan


@beartype
def _set_horn(
    parameter: str,
    partial_derivatives: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    p, q = partial_derivatives

    if parameter == "Slope":
        out_array = slope(p, q)
    elif parameter == "Aspect":
        out_array = aspect(p, q, "Horn")

    return out_array


@beartype
def set_horn(
    raster: rasterio.io.DatasetReader,
    parameters: Sequence[Literal["Slope", "Aspect"]],
    scaling_factor: Optional[Number] = 1,
    min_slope: Optional[Number] = 0,
    slope_unit: Optional[Literal["radians", "degrees", "rise"]] = "radians",
    aspect_unit: Optional[Literal["radians", "degrees"]] = "radians",
) -> dict:
    """Calculate Slope and Aspect after Horn (1981).

    Reference:
    Horn, B.K., 1981: Hill shading and the reflectance map, Proceedings of the IEEE, 69/1: 14-47.

    Args:
        raster: The input raster data.
        scaling_factor: Factor to modify values, e.g. for unit conversion.
        parameter: Parameter to be calculated.
        gradient_unit: Valid values for slope are [radians, degrees, rise] and for aspect [radians, degrees].
        min_slope: Minimum slope gradient in degrees below a surface is treated as flat surface.

    Returns:
        Selected surface attribute and respective updated metadata.

        Slope is in range [0, 90] degrees.
        Aspect is in range [0, 360] degrees with North (0/360), East (90), South (180) and West (270).

    Raises:
        InvalidRasterBandException: Raster has more than one band.
        NonSquarePixelSizeException: Pixel dimensions do not have same length.
        InvalidParameterValueException: Wrong input parameters provided.
    """
    if raster.count > 1:
        raise InvalidRasterBandException("Only one-band raster supported.")

    if check_quadratic_pixels(raster) is False:
        raise NonSquarePixelSizeException("Processing requires quadratic pixel dimensions.")

    if scaling_factor <= 0:
        raise InvalidParameterValueException("Scaling factor must be greater than 0.")

    if min_slope < 0:
        raise NumericValueSignException("Minimum slope must be a positive number.")

    surface_array = raster.read()
    surface_array = reduce_ndim(surface_array)
    surface_array = nodata_to_nan(surface_array, nodata_value=raster.nodata)
    surface_array = scale_raster(surface_array, scaling_factor)

    p, q, _, _, _ = coefficients(surface_array, cellsize=raster.res[0], method="Horn", coefficients=["p", "q"])
    gradient = slope(p, q) if min_slope > 0 else (p, q)

    out_dict = {}
    for parameter in parameters:
        out_array = (
            gradient if parameter == "Slope" and isinstance(gradient, np.ndarray) else _set_horn(parameter, (p, q))
        )

        out_array = convert_rad_to_deg(out_array) if parameter == "Slope" and slope_unit == "degrees" else out_array
        out_array = convert_rad_to_rise(out_array) if parameter == "Slope" and slope_unit == "rise" else out_array
        out_array = convert_rad_to_deg(out_array) if parameter == "Aspect" and aspect_unit == "degrees" else out_array

        out_array = (
            set_flat_pixels(out_array, gradient, np.radians(min_slope), parameter)
            if parameter != "Slope"
            else out_array
        )

        out_nodata = -9999
        out_array = nan_to_nodata(out_array, nodata_value=out_nodata).astype(np.float32)
        out_meta = raster.meta.copy()
        out_meta.update({"dtype": out_array.dtype.name, "nodata": out_nodata})

        out_dict[parameter] = (out_array, out_meta)

    return out_dict
