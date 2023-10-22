import numpy as np
import rasterio
from numbers import Number
from beartype import beartype
from beartype.typing import Literal

from eis_toolkit.exceptions import (
    InvalidRasterBandException,
    InvalidParameterValueException,
)


@beartype
def slope(
    p: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """Calculate the slope gradient of a given surface."""

    return np.arctan(np.sqrt(p**2 + q**2))


@beartype
def aspect(
    p: np.ndarray,
    q: np.ndarray,
    method: str,
) -> np.ndarray:
    """Calculate the slope direction of a given surface."""

    if method == "Horn":
        out_array = np.pi + np.arctan2(p, -q)
    elif method == "ZevenbergenThorne":
        out_array = np.pi + np.arctan2(p, q)

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
    aspect = np.squeeze(aspect) if aspect.ndim >= 3 else aspect
    out_nodata = -9999

    mask_nd = np.equal(aspect, -1)
    mask_nodata = np.equal(aspect, raster.nodata)

    if np.issubdtype(aspect.dtype, np.integer):
        aspect = aspect.astype(float)

    aspect = np.where(np.logical_or(mask_nd, mask_nodata), np.nan, aspect)
    aspect = np.radians(aspect) if unit == "degrees" else aspect

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
    out_array = np.where(mask_nodata, out_nodata, out_array)
    out_array = out_array.astype(np.result_type(np.int8, out_nodata))

    out_mapping = {direction: i + 1 for i, direction in enumerate(dir_classes[:num_classes])}
    out_mapping["ND"] = -1

    out_meta = raster.meta.copy()
    out_meta.update({"dtype": out_array.dtype.name, "nodata": out_nodata})

    return out_array, out_mapping, out_meta


@beartype
def classify_aspect(
    raster: rasterio.io.DatasetReader,
    unit: Literal["radians", "degrees"] = "radians",
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
