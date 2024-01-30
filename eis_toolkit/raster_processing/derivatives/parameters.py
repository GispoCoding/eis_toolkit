from numbers import Number

import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Literal, Optional, Sequence

from eis_toolkit.exceptions import (
    InvalidParameterValueException,
    InvalidRasterBandException,
    NonSquarePixelSizeException,
)
from eis_toolkit.raster_processing.derivatives.partial_derivatives import _coefficients
from eis_toolkit.raster_processing.derivatives.utilities import _scale_raster, _set_flat_pixels
from eis_toolkit.utilities.checks.raster import check_quadratic_pixels
from eis_toolkit.utilities.conversions import convert_rad_to_deg, convert_rad_to_rise
from eis_toolkit.utilities.miscellaneous import reduce_ndim
from eis_toolkit.utilities.nodata import nan_to_nodata, nodata_to_nan


def _divide(
    numerator: np.ndarray,
    denumerator: np.ndarray,
) -> np.ndarray:
    """
    Safely divide two arrays.

    Args:
        numerator: The numerator array.
        denumerator: The denumerator array.

    Returns:
        The result of dividing the numerator by the denumerator.
    """
    minimum = 1e-6
    return numerator / np.where(denumerator < minimum, minimum, denumerator)


def _first_order(
    parameter: str,
    coefficients: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """
    Calculate the first order surface attributes slope (gradient) and aspect (slope direction).

    Args:
        parameter: The surface parameter to calculate.
        coefficients: The coefficients used in the calculation.

    Returns:
        The calculated surface attribute.
    """
    p, q = coefficients

    if parameter == "G":
        return np.arctan(np.sqrt(p**2 + q**2))
    elif parameter == "A":
        return np.pi + np.arctan2(p, q)


def _second_order_basic_set(
    parameter: str,
    coefficients: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Calculate the second order surface attributes (curvatures).

    Args:
        parameter: The surface parameter to calculate.
        coefficients: The coefficients used in the calculation.

    Returns:
        The calculated surface attribute.
    """
    p, q, r, s, t = coefficients

    if parameter == "planc":
        return _divide(-2 * (q**2 * r - p * q * s + p**2 * t), np.sqrt(np.power(p**2 + q**2, 3)))
    elif parameter == "profc":
        return _divide(
            -2 * (p**2 * r + p * q * s + q**2 * t), (p**2 + q**2) * (np.sqrt(np.power(1 + p**2 + q**2, 3)))
        )
    elif parameter == "profc_min":
        return -r - t - np.sqrt(np.square(r - t) + s**2)
    elif parameter == "profc_max":
        return -r - t + np.sqrt(np.square(r - t) + s**2)
    elif parameter == "longc":
        return _divide(-2 * (p**2 * r + p * q * s + q**2 * t), p**2 + q**2)
    elif parameter == "crosc":
        return _divide(-2 * (q**2 * r - p * q * s + p**2 * t), p**2 + q**2)
    elif parameter == "rot":
        return _divide((p**2 - q**2) * s - p * q * (r - t), np.sqrt(np.power(p**2 + q**2, 3)))
    elif parameter == "K":
        return (r * t - s**2) / np.square(1 + p**2 + q**2)
    elif parameter == "genc":
        return -2 * (r + t)
    elif parameter == "tangc":
        return _divide(-2 * (q**2 * r - p * q * s + p**2 * t), (p**2 + q**2) * np.sqrt(1 + p**2 + q**2))


@beartype
def first_order(
    raster: rasterio.io.DatasetReader,
    parameters: Sequence[Literal["G", "A"]],
    scaling_factor: Optional[Number] = 1,
    slope_tolerance: Optional[Number] = 0,
    slope_gradient_unit: Literal["degrees", "radians", "rise"] = "radians",
    slope_direction_unit: Literal["degrees", "radians"] = "radians",
    method: Literal["Horn", "Evans", "Young", "Zevenbergen"] = "Horn",
) -> dict:
    """Calculate the first order surface attributes.

    For compatibility for slope and aspect calculations with ArcGIS or QGIS, choose Method Horn (1981).

    Args:
        raster: Input raster.
        parameters: List of surface parameters to be calculated.
        scaling_factor: Scaling factor to be applied to the raster data set. Default to 1.
        slope_tolerance: Tolerance value for flat pixels. Default to 0.
        slope_gradient_unit: Unit of the slope gradient parameter. Default to radians.
        slope_direction_unit: Unit of the slope direction parameter. Default to radians.
        method: Method for calculating the coefficients. Default to the Horn (1981) method.

    Returns:
        Selected surface attributes and respective updated metadata.

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
        raise InvalidParameterValueException("Value must be greater than 0.")

    raster_array = raster.read()
    raster_array = reduce_ndim(raster_array)
    raster_array = nodata_to_nan(raster_array, nodata_value=raster.nodata)
    raster_array = _scale_raster(raster_array, scaling_factor)

    cellsize = raster.res[0]
    p, q, *_ = _coefficients(raster_array, cellsize, method)
    q = -q if method == "Horn" else q

    slope_gradient = _first_order("G", (p, q)) if slope_tolerance > 0 else (p, q)

    out_dict = {}
    out_nodata = -9999
    for parameter in parameters:
        out_array = (
            slope_gradient
            if parameter == "G" and isinstance(slope_gradient, np.ndarray)
            else _first_order(parameter, (p, q))
        )

        if (parameter == "G" and slope_gradient_unit == "degrees") or (
            parameter == "A" and slope_direction_unit == "degrees"
        ):
            out_array = convert_rad_to_deg(out_array)
        elif parameter == "G" and slope_gradient_unit == "rise":
            out_array = convert_rad_to_rise(out_array)

        out_array = (
            _set_flat_pixels(out_array, slope_gradient, slope_tolerance, parameter) if parameter != "G" else out_array
        )

        out_array = nan_to_nodata(out_array, nodata_value=out_nodata).astype(np.float32)
        out_meta = raster.meta.copy()
        out_meta.update({"dtype": out_array.dtype.name, "nodata": out_nodata})
        out_dict[parameter] = (out_array, out_meta)

    return out_dict


@beartype
def second_order_basic_set(
    raster: rasterio.io.DatasetReader,
    parameters: Sequence[
        Literal[
            "planc",
            "profc",
            "profc_min",
            "profc_max",
            "longc",
            "crosc",
            "rot",
            "K",
            "genc",
            "tangc",
        ]
    ],
    scaling_factor: Optional[Number] = 1,
    slope_tolerance: Optional[Number] = 0,
    method: Literal["Evans", "Young", "Zevenbergen"] = "Young",
) -> dict:
    """Calculate the second order surface attributes.

    References:
        Young, M., 1978: Terrain analysis program documentation. Report 5 on Grant DA-ERO-591-73-G0040,
        'Statistical characterization of altitude matrices by computer'. Department of Geography,
        University of Durham, England: 27 pp.

        Zevenbergen, L.W. and Thorne, C.R., 1987: Quantitative analysis of land surface topography,
        Earth Surface Processes and Landforms, 12: 47-56.

        Wood, J., 1996: The Geomorphological Characterisation of Digital Elevation Models. Doctoral Thesis.
        Department of Geography, University of Leicester, England: 466 pp.

        Parameters longc and crosc from are referenced by Zevenbergen & Thorne (1987) as profile and plan curvature.
        For compatibility with ArcGIS, choose Method Zevenbergen & Thorne (1987) and parameters longc and crosc.

    Args:
        raster: Input raster.
        parameters: List of surface parameters to be calculated.
        scaling_factor: Scaling factor to be applied to the raster data set. Default to 1.
        slope_tolerance: Tolerance value for flat pixels. Default to 0.
        method: Method for calculating the coefficients. Default to the Young (1978) method.

    Returns:
        Selected surface attributes and respective updated metadata.

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
        raise InvalidParameterValueException("Value must be greater than 0.")

    raster_array = raster.read()
    raster_array = reduce_ndim(raster_array)
    raster_array = nodata_to_nan(raster_array, nodata_value=raster.nodata)
    raster_array = _scale_raster(raster_array, scaling_factor)

    cellsize = raster.res[0]
    p, q, r, s, t = _coefficients(raster_array, cellsize, method)
    slope_gradient = _first_order("G", (p, q)) if slope_tolerance > 0 else (p, q)

    out_dict = {}
    out_nodata = -9999
    for parameter in parameters:
        out_array = _second_order_basic_set(parameter, (p, q, r, s, t))
        out_array = _set_flat_pixels(out_array, slope_gradient, slope_tolerance, parameter)
        out_array = nan_to_nodata(out_array, nodata_value=out_nodata).astype(np.float32)
        out_meta = raster.meta.copy()
        out_meta.update({"dtype": out_array.dtype.name, "nodata": out_nodata})
        out_dict[parameter] = (out_array, out_meta)

    return out_dict
