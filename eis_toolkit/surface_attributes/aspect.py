from numbers import Number

import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Literal, Optional

from eis_toolkit.checks.raster import check_quadratic_pixels
from eis_toolkit.exceptions import (
    InvalidRasterBandException,
    NonSquarePixelSizeException,
    InvalidParameterValueException,
)
from eis_toolkit.surface_attributes.partial_derivatives import _method_horn
from eis_toolkit.surface_attributes.slope import _get_slope
from eis_toolkit.utilities.nodata import nan_to_nodata, nodata_to_nan


@beartype
def _get_aspect(
    raster: rasterio.io.DatasetReader,
    method: Literal["Horn81"],
) -> tuple[np.ndarray, dict]:

    cellsize = raster.res[0]
    out_meta = raster.meta.copy()

    out_array = raster.read()
    if out_array.ndim >= 3:
        out_array = np.squeeze(out_array)

    out_array = nodata_to_nan(out_array, nodata_value=raster.nodata)

    if method == "Horn81":
        p = _method_horn(out_array, cellsize=cellsize, parameter="p")
        q = _method_horn(out_array, cellsize=cellsize, parameter="q")

    out_array = (np.pi + np.arctan2(p, -q)) * (180.0 / np.pi)
    out_array = np.where(np.logical_and(p == 0, q == 0), -1, out_array)

    out_array = nan_to_nodata(out_array, nodata_value=raster.nodata).astype(np.float16)
    out_meta.update({"dtype": out_array.dtype.name})

    return out_array, out_meta


@beartype
def _mask_aspect(
    raster: rasterio.io.DatasetReader,
    method: Literal["Horn81"],
    scaling_factor: Number,
    min_slope: Number,
    aspect: np.ndarray,
) -> np.ndarray:

    slope = _get_slope(raster, method, scaling_factor)

    out_array = aspect
    out_array = nodata_to_nan(out_array, nodata_value=raster.nodata)
    out_array = np.where(slope < min_slope, -1, out_array)
    out_array = nan_to_nodata(out_array, nodata_value=raster.nodata)

    return out_array


@beartype
def _classify_aspect(
    raster: rasterio.io.DatasetReader,
    unit: Literal["degree", "radians"],
    num_classes: Number,
) -> tuple[np.ndarray, dict, dict]:

    aspect = raster.read()
    nodata = int(raster.nodata) if np.can_cast(raster.nodata, np.integer) else raster.nodata

    if aspect.ndim >= 3:
        aspect = np.squeeze(aspect)

    mask_nd = np.equal(aspect, -1)
    mask_nodata = np.equal(aspect, nodata)

    if np.issubdtype(aspect.dtype, np.integer):
        aspect = aspect.astype(float)

    aspect = np.where(np.logical_or(mask_nd, mask_nodata), np.nan)

    if unit == "degree":
        aspect = np.radians(aspect)

    # Adjust the array to rotate 22.5 degrees counter-clockwise
    aspect = (aspect + np.pi / num_classes) % (2 * np.pi)

    if num_classes == 8:
        dir_classes = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    elif num_classes == 16:
        dir_classes = np.array(
            ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        )

    dir_classes = np.append(dir_classes, ["ND", "NODATA"])

    # Determine index to each value in the aspect array
    out_array = np.digitize(aspect, np.linspace(0, 2 * np.pi, num_classes + 1))
    out_array = np.where(mask_nd, -1, out_array)
    out_array = np.where(mask_nodata, nodata, out_array)
    out_array = np.astype(np.result_type(np.int8, nodata))

    out_mapping = {direction: i + 1 for i, direction in enumerate(dir_classes)}

    out_meta = raster.meta.copy()
    out_meta["dtype"] = out_array.dtype.name

    return out_array, out_mapping, out_meta


@beartype
def get_aspect(
    raster: rasterio.io.DatasetReader,
    method: Literal["Horn81"] = "Horn81",
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
        scaling_factor: Factor to modify values, e.g. for unit conversion.
        min_slope: Slope value in degree below a cell will be considered as flat surface.

    Returns:
        The calculated aspect in degree (0-360).
    """

    if raster.count > 1:
        raise InvalidRasterBandException("Only one-band raster supported.")

    if check_quadratic_pixels(raster) is False:
        raise NonSquarePixelSizeException("Processing requires quadratic pixel dimensions.")

    out_array, out_meta = _get_aspect(raster, method)

    if min_slope is not None:
        out_array = _mask_aspect(raster, method, scaling_factor, min_slope, aspect=out_array)

    return out_array, out_meta


@beartype
def classify_aspect(
    raster: rasterio.io.DatasetReader,
    unit: Literal["degree", "radians"] = "degree",
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
        unit: The unit of the input raster. Either 'degree' or 'radians'
        num_classes: The number of classes for discretization. Either 8 or 16 classes allowed.

    Returns:
        The classified aspect raster, a class mapping dictionary and the updated metadata.
    """

    if raster.count > 1:
        raise InvalidRasterBandException("Only one-band raster supported.")

    if unit != "degree" and unit != "radians":
        raise InvalidParameterValueException("Only 'degree' or 'radians' units allowed.")

    if num_classes != 8 and num_classes != 16:
        raise InvalidParameterValueException("Only 8 or 16 classes allowed for classification!")

    return _classify_aspect(raster, unit, num_classes)
