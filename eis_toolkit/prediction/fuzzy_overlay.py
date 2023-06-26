import numpy as np
from beartype import beartype

from eis_toolkit.exceptions import InvalidParameterValueException


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
    if any(band_data.min() < 0 or band_data.max() > 1 for band_data in data):
        raise InvalidParameterValueException("All data must be in range [0, 1]")

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
    if any(band_data.min() < 0 or band_data.max() > 1 for band_data in data):
        raise InvalidParameterValueException("All data must be in range [0, 1]")

    return data.max(axis=0)


@beartype
def product_overlay(data: np.ndarray) -> np.ndarray:
    """Compute an 'product' overlay operation with fuzzy logic.

    Args:
        data: The input data as a 3D Numpy array. Each 2D array represents a raster band.
            Data points should be in the range [0, 1].

    Returns:
        2D Numpy array with the result of the 'product' overlay operation. Values are in range [0, 1].

    Raises:
        InvalidParameterValueException: If data values are not in range [0, 1].
    """
    if any(band_data.min() < 0 or band_data.max() > 1 for band_data in data):
        raise InvalidParameterValueException("All data must be in range [0, 1]")

    return np.prod(data, axis=0)


@beartype
def sum_overlay(data: np.ndarray) -> np.ndarray:
    """Compute an 'sum' overlay operation with fuzzy logic.

    Args:
        data: The input data as a 3D Numpy array. Each 2D array represents a raster band.
            Data points should be in the range [0, 1].

    Returns:
        2D Numpy array with the result of the 'sum' overlay operation. Values are in range [0, 1].

    Raises:
        InvalidParameterValueException: If data values are not in range [0, 1].
    """
    if any(band_data.min() < 0 or band_data.max() > 1 for band_data in data):
        raise InvalidParameterValueException("All data must be in range [0, 1]")

    return data.sum(axis=0) - np.prod(data, axis=0)


@beartype
def gamma_overlay(data: np.ndarray, gamma: float) -> np.ndarray:
    """Compute an 'gamma' overlay operation with fuzzy logic.

    Args:
        data: The input data as a 3D Numpy array. Each 2D array represents a raster band.
            Data points should be in the range [0, 1].
        gamma: The gamma parameter. With gamma value 0, result will be same as 'product'overlay.
            When gamma is closer to 1, the weight of 'sum' overlay is increased.
            Value must be in the range [0, 1].

    Returns:
        2D Numpy array with the result of the 'gamma' overlay operation. Values are in range [0, 1].

    Raises:
        InvalidParameterValueException: If data values or gamma are not in range [0, 1].
    """
    if any(band_data.min() < 0 or band_data.max() > 1 for band_data in data):
        raise InvalidParameterValueException("All data must be in range [0, 1]")

    if gamma < 0 or gamma > 1:
        raise InvalidParameterValueException("The gamma parameter must be in range [0, 1]")

    fuzzy_sum = data.sum(axis=0) - np.prod(data, axis=0)
    fuzzy_product = np.prod(data, axis=0)
    return fuzzy_product ** (1 - gamma) * fuzzy_sum**gamma
