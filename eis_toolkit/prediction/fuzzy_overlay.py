import numpy as np
from beartype import beartype

from eis_toolkit import exceptions


def _check_input_data(data: np.ndarray):
    if data.min() < 0 or data.max() > 1:
        raise exceptions.InvalidParameterValueException("All data must be in range [0, 1].")
    if data.ndim != 3:
        raise exceptions.InvalidParameterValueException("Input data for overlay should be 3D numpy array.")


@beartype
def and_overlay(data: np.ndarray) -> np.ndarray:
    """Compute an 'and' overlay operation with fuzzy logic.

    Args:
        data: The input data as a 3D Numpy array. Each 2D array represents a raster band.
            Data points should be in the range [0, 1].

    Returns:
        2D Numpy array with the result of the 'and' overlay operation. Values are in range [0, 1].

    Raises:
        InvalidParameterValueException: If data values are not in range [0, 1].
    """
    _check_input_data(data=data)

    return data.min(axis=0)


@beartype
def or_overlay(data: np.ndarray) -> np.ndarray:
    """Compute an 'or' overlay operation with fuzzy logic.

    Args:
        data: The input data as a 3D Numpy array. Each 2D array represents a raster band.
            Data points should be in the range [0, 1].

    Returns:
        2D Numpy array with the result of the 'or' overlay operation. Values are in range [0, 1].

    Raises:
        InvalidParameterValueException: If data values are not in range [0, 1].
    """
    _check_input_data(data=data)

    return data.max(axis=0)


@beartype
def product_overlay(data: np.ndarray) -> np.ndarray:
    """Compute a 'product' overlay operation with fuzzy logic.

    Args:
        data: The input data as a 3D Numpy array. Each 2D array represents a raster band.
            Data points should be in the range [0, 1].

    Returns:
        2D Numpy array with the result of the 'product' overlay operation. Values are in range [0, 1].

    Raises:
        InvalidParameterValueException: If data values are not in range [0, 1].
    """
    _check_input_data(data=data)

    return np.prod(data, axis=0)


@beartype
def sum_overlay(data: np.ndarray) -> np.ndarray:
    """Compute a 'sum' overlay operation with fuzzy logic.

    Args:
        data: The input data as a 3D Numpy array. Each 2D array represents a raster band.
            Data points should be in the range [0, 1].

    Returns:
        2D Numpy array with the result of the 'sum' overlay operation. Values are in range [0, 1].

    Raises:
        InvalidParameterValueException: If data values are not in range [0, 1].
    """
    _check_input_data(data=data)

    return data.sum(axis=0) - np.prod(data, axis=0)


@beartype
def gamma_overlay(data: np.ndarray, gamma: float) -> np.ndarray:
    """Compute a 'gamma' overlay operation with fuzzy logic.

    Args:
        data: The input data as a 3D Numpy array. Each 2D array represents a raster band.
            Data points should be in the range [0, 1].
        gamma: The gamma parameter. With gamma value of 0, the result will be the same as 'product' overlay.
            When gamma is closer to 1, the weight of the 'sum' overlay is increased.
            Value must be in the range [0, 1].

    Returns:
        2D Numpy array with the result of the 'gamma' overlay operation. Values are in range [0, 1].

    Raises:
        InvalidParameterValueException: If data values or gamma are not in range [0, 1].
    """
    if gamma < 0 or gamma > 1:
        raise exceptions.InvalidParameterValueException("The gamma parameter must be in range [0, 1]")

    sum = sum_overlay(data=data)
    product = product_overlay(data=data)
    return product ** (1 - gamma) * sum**gamma
