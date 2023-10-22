import numpy as np
import rasterio
from numbers import Number
from beartype import beartype
from beartype.typing import Literal, Optional, Union

from eis_toolkit.checks.raster import check_quadratic_pixels
from eis_toolkit.exceptions import (
    InvalidRasterBandException,
    NonSquarePixelSizeException,
    InvalidParameterValueException,
)
from eis_toolkit.surface_attributes.partial_derivatives import coefficients
from eis_toolkit.surface_attributes.utilities import set_flat_pixels
from eis_toolkit.utilities.conversions import convert_rad_to_rise, convert_rad_to_deg
from eis_toolkit.utilities.nodata import nan_to_nodata, nodata_to_nan


@beartype
def _slope(
    p: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """Calculate the slope of a given surface."""

    return np.arctan(np.sqrt(p**p + q**q))


@beartype
def _aspect(
    p: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """Calculate the slope direction of a given surface."""

    return np.pi + np.arctan2(p, q)


@beartype
def _planc(
    p: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    s: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Calculate the plan curvature of a given surface."""

    return -(t * p**p + r * q**q - 2 * s * p * q) / np.sqrt(np.power(p**p + q**q, 3))


@beartype
def _profc(
    p: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    s: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Calculate the profile curvature of a given surface."""

    return -(r * p**p + t * q**q + 2 * s * p * q) / ((p**p + q**q) * np.sqrt(np.power(p**p + q**q, 3)))


@beartype
def _set_zevenbergen_thorne(
    raster: rasterio.io.DatasetReader,
    parameter: str,
    scaling_factor: Number,
    min_slope: Number,
    gradient_unit: str,
    partial_derivatives: Optional[
        tuple[
            Union[np.ndarray, None],
            Union[np.ndarray, None],
            Union[np.ndarray, None],
            Union[np.ndarray, None],
            Union[np.ndarray, None],
        ]
    ] = None,
) -> tuple[np.ndarray, dict]:
    out_nodata = -9999

    if partial_derivatives is None:
        raster_array = raster.read()
        raster_array = np.squeeze(raster_array) if raster_array.ndim >= 3 else raster_array
        raster_array = nodata_to_nan(raster_array, nodata_value=raster.nodata)
        raster_array = raster_array * scaling_factor if scaling_factor != 1 else raster_array

        if parameter == "Slope" or parameter == "Aspect":
            coeff_selection = ["p", "q"]
        else:
            coeff_selection = ["p", "q", "r", "s", "t"]

        p, q, r, s, t = coefficients(
            raster_array, cellsize=raster.res[0], method="ZevenbergenThorne", coefficients=coeff_selection
        )
    else:
        p, q, r, s, t = partial_derivatives

    if parameter == "Slope" or min_slope > 0:
        slope_array = _slope(p, q)

    if parameter == "Slope":
        out_array = slope_array
        out_array = convert_rad_to_deg(out_array) if gradient_unit == "degrees" else out_array
        out_array = convert_rad_to_rise(out_array) if gradient_unit == "rise" else out_array
    elif parameter == "Aspect":
        out_array = _aspect(p, q)

        if min_slope > 0:
            out_array = set_flat_pixels(out_array, _slope(p, q), np.radians(min_slope), "Aspect")
        else:
            out_array = np.where(np.logical_and(p == 0, q == 0), -1, out_array)

        out_array = convert_rad_to_deg(out_array) if gradient_unit == "degrees" else out_array
    else:
        if parameter == "PlanC":
            out_array = _planc(p, q, r, s, t)
        elif parameter == "ProfC":
            out_array = _profc(p, q, r, s, t)

        if min_slope > 0:
            out_array = set_flat_pixels(out_array, slope_array, np.radians(min_slope), "Curvature")

    out_array = nan_to_nodata(out_array, nodata_value=out_nodata).astype(np.float32)

    out_meta = raster.meta.copy()
    out_meta.update({"dtype": out_array.dtype.name, "nodata": out_nodata})

    return out_array, out_meta


@beartype
def set_zevenbergen_thorne(
    raster: rasterio.io.DatasetReader,
    parameter: Literal["Slope", "Aspect", "PlanC", "ProfC"],
    scaling_factor: Number = 1,
    min_slope: Number = 0,
    gradient_unit: Optional[Literal["radians", "degrees", "rise"]] = "radians",
) -> tuple[np.ndarray, dict]:
    """Calculate the partial derivatives of a surface after Zevenbergen & Thorne (1987).

    Reference:
    Zevenbergen, L.W. and Thorne, C.R., 1987: Quantitative analysis of land surface topography,
    Earth Surface Processes and Landforms, 12: 47-56.

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
        Curvatures [-inf, 0] describe concave and [0, inf] are convex surfaces. A value of 0 describes a flat surface.

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

    if gradient_unit == "rise" and parameter == "Aspect":
        raise InvalidParameterValueException("Rise is not a valid unit for aspect.")

    out_array, out_meta = _set_zevenbergen_thorne(raster, parameter, scaling_factor, min_slope, gradient_unit)

    return out_array, out_meta
