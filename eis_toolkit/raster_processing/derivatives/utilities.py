from numbers import Number

import numpy as np
from beartype import beartype
from beartype.typing import Union


@beartype
def _scale_raster(
    data: np.ndarray,
    scaling_factor: Number,
) -> np.ndarray:
    """
    Scale raster data by a given factor.

    Args:
        data: The input raster data as a numpy array.
        scaling_factor: The scaling factor to apply.

    Returns:
        The scaled raster data.
    """
    return data * scaling_factor if scaling_factor != 1 else data


@beartype
def _set_flat_pixels(
    in_array: np.ndarray,
    slope_gradient: Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
    slope_tolerance: Number,
    parameter: str,
) -> np.ndarray:
    """Treating values below a certain gradient as flat surface.

    Args:
        in_array: Input array.
        slope_gradient: Input slope array in degrees or tuple of partial derivatives p and q.
        slope_tolerance: Value below a surface will be treated as flat surface (degrees).
        parameter: The surface attribute to modify.

    Returns:
        Array of the modified surface attribute.
    """
    replacement_value = -1 if parameter == "A" else 0

    if slope_tolerance == 0:
        p, q = slope_gradient
        return np.where(np.logical_and(p == 0, q == 0), replacement_value, in_array)
    elif slope_tolerance > 0:
        return np.where(slope_gradient <= np.radians(slope_tolerance), replacement_value, in_array)
