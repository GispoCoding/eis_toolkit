from numbers import Number

import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Literal, Optional

from eis_toolkit.checks.raster import check_quadratic_pixels
from eis_toolkit.exceptions import (
    InvalidParameterValueException,
    InvalidRasterBandException,
    NonSquarePixelSizeException,
)
from eis_toolkit.surface_attributes.partial_derivatives import _method_horn
from eis_toolkit.surface_attributes.slope import _get_slope
from eis_toolkit.utilities.nodata import nan_to_nodata, nodata_to_nan


@beartype
def _get_aspect(
    raster: rasterio.io.DatasetReader,
    method: Literal["Horn81"],
    unit: Literal["degrees", "radians"],
) -> tuple[np.ndarray, dict]:

    cellsize = raster.res[0]
    out_meta = raster.meta.copy()
    nodata = -9999

    out_array = raster.read()
    if out_array.ndim >= 3:
        out_array = np.squeeze(out_array)

    out_array = nodata_to_nan(out_array, nodata_value=raster.nodata)

    if method == "Horn81":
        p = _method_horn(out_array, cellsize=cellsize, parameter="p")
        q = _method_horn(out_array, cellsize=cellsize, parameter="q")

    out_array = np.pi + np.arctan2(p, -q)
    out_array = np.degrees(out_array) if unit == "degrees" else out_array
    out_array = np.where(np.logical_and(p == 0, q == 0), -1, out_array)
    out_array = nan_to_nodata(out_array, nodata_value=nodata).astype(np.float32)

    out_meta.update({"dtype": out_array.dtype.name, "nodata": nodata})

    return out_array, out_meta


@beartype
def _mask_aspect(
    raster: rasterio.io.DatasetReader,
    method: Literal["Horn81"],
    min_slope: Number,
    scaling_factor: Number,
    aspect: np.ndarray,
) -> np.ndarray:
    slope, _ = _get_slope(
        raster,
        method,
        scaling_factor,
        unit="degrees",
    )

    out_array = aspect
    out_array = nodata_to_nan(out_array, nodata_value=raster.nodata)
    out_array = np.where(np.logical_and(slope > 0, slope < min_slope), -1, out_array)
    out_array = nan_to_nodata(out_array, nodata_value=raster.nodata)

    return out_array


@beartype
def _classify_aspect(
    raster: rasterio.io.DatasetReader,
    unit: Literal["degrees", "radians"],
    num_classes: Number,
) -> tuple[np.ndarray, dict, dict]:
    """
    Classify a provided aspect raster into 8 or 16 directions.

    Directions and interval for 8 classes:
    N: (337.5, 22.5), NE: (22.5, 67.5),
    E: (67.5, 112.5), SE: (112.5, 157.5),
    S: (157.5, 202.5), SW: (202.5, 247.5),
    W: (247.5, 292.5), NW: (292.5, 337.5)

    Directions and interval for 16 classes:
    N: (348.75, 11.25), NNE: (11.25, 33.75), NE: (33.75, 56.25), ENE: (56.25, 78.75),
    E: (78.75, 101.25), ESE: (101.25, 123.75), SE: (123.75, 146.25), SSE: (146.25, 168.75),
    S: (168.75, 191.25), SSW: (191.25, 213.75), SW: (213.75, 236.25), WSW: (236.25, 258.75),
    W: (258.75, 281.25), WNW: (281.25, 303.75), NW: (303.75, 326.25), NNW: (326.25, 348.75)

    Flat pixels (input value: -1) will be kept, the class is called ND (not defined).

    Args:
        raster: The input aspect raster data.
        unit: Corresponding unit of the input aspect raster, either "degrees" or "radians"
        num_classes: The desired number of classes for classification, either 8 or 16.

    Returns:
        The classified aspect raster with integer values for each class.
        The mapping for the corresponding direction for each class.
        The updated raster meta data.
    """

    aspect = raster.read()
    nodata = -9999

    if aspect.ndim >= 3:
        aspect = np.squeeze(aspect)

    mask_nd = np.equal(aspect, -1)
    mask_nodata = np.equal(aspect, nodata)

    if np.issubdtype(aspect.dtype, np.integer):
        aspect = aspect.astype(float)

    aspect = np.where(np.logical_or(mask_nd, mask_nodata), np.nan, aspect)

    if unit == "degrees":
        aspect = np.radians(aspect)

    # Adjust the array to rotate 22.5 degrees counter-clockwise
    aspect = (aspect + np.pi / num_classes) % (2 * np.pi)

    if num_classes == 8:
        dir_classes = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW", "ND"])
    elif num_classes == 16:
        dir_classes = np.array(
            ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "ND"]
        )

    # Determine index to each value in the aspect array
    out_array = np.digitize(aspect, np.linspace(0, 2 * np.pi, num_classes + 1))
    out_array = np.where(mask_nd, -1, out_array)
    out_array = np.where(mask_nodata, nodata, out_array)
    out_array = out_array.astype(np.result_type(np.int8, nodata))

    out_mapping = {direction: i + 1 for i, direction in enumerate(dir_classes[:num_classes])}
    out_mapping["ND"] = -1

    out_meta = raster.meta.copy()
    out_meta.update({"dtype": out_array.dtype.name, "nodata": nodata})

    return out_array, out_mapping, out_meta


@beartype
def get_aspect(
    raster: rasterio.io.DatasetReader,
    method: Literal["Horn81"] = "Horn81",
    unit: Literal["degrees", "radians"] = "degrees",
    scaling_factor: Optional[Number] = 1,
    min_slope: Optional[Number] = None,
) -> tuple[np.ndarray, dict]:
    """
    Calculate the aspect of a given surface.

    A raster cell with a slope of 0 essentially implies a flat surface
    with no inclination in any particular direction. Thus, for cells with a
    slope of 0, the aspect is not defined and set to -1.

    Args:
        raster: The input raster data.
        method: Basic method used to calculate partial derivatives.
        unit: Unit for aspect output raster. Either "degrees" or "radians".
        scaling_factor: Factor to modify values, e.g. for unit conversion.
        min_slope: Slope value in degree below a cell will be considered as flat surface.

    Returns:
        The calculated aspect in degree [0, 360] or radians [0, 2*pi].
    """

    if raster.count > 1:
        raise InvalidRasterBandException("Only one-band raster supported.")

    if check_quadratic_pixels(raster) is False:
        raise NonSquarePixelSizeException("Processing requires quadratic pixel dimensions.")

    out_array, out_meta = _get_aspect(raster, method, unit)

    if min_slope is not None:
        out_array = _mask_aspect(raster, method, min_slope, scaling_factor, aspect=out_array)

    return out_array, out_meta


@beartype
def classify_aspect(
    raster: rasterio.io.DatasetReader,
    unit: Literal["degrees", "radians"] = "degrees",
    num_classes: int = 8,
) -> tuple[np.ndarray, dict, dict]:
    """
    Classify an aspect raster data set.

    Can classify an aspect raster into 8 or 16 equally spaced directions with
    intervals of pi/4 and pi/8, respectively.

    Exemplary for 8 classes, the center of the intervall for North direction is 0°/360°
    and edges are [337.5°, 22.5°], counting forward in clockwise direction. For 16 classes,
    the intervall-width is half with edges at [348,75°, 11,25°].

    The method considers both flat pixels (aspect of -1) and raster.nodata values as separate classes.

    Args:
        raster: The input raster data.
        unit: The unit of the input raster. Either "degrees" or "radians"
        num_classes: The number of classes for discretization. Either 8 or 16 classes allowed.

    Returns:
        The classified aspect raster, a class mapping dictionary and the updated metadata.
    """

    if raster.count > 1:
        raise InvalidRasterBandException("Only one-band raster supported.")

    if num_classes != 8 and num_classes != 16:
        raise InvalidParameterValueException("Only 8 or 16 classes allowed for classification!")

    return _classify_aspect(raster, unit, num_classes)
