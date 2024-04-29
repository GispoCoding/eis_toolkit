import numpy as np
from beartype import beartype
from beartype.typing import Sequence, Union

from eis_toolkit.exceptions import InvalidDatasetException, InvalidParameterValueException
from eis_toolkit.utilities.raster import stack_raster_arrays


def _prepare_data_for_fuzzy_overlay(data: Union[Sequence[np.ndarray], np.ndarray]) -> np.ndarray:
    if isinstance(data, Sequence):
        data = stack_raster_arrays(data)
    if (data.ndim == 3 and data.shape[0] < 2) or data.ndim == 2:
        raise InvalidDatasetException("At least 2 arrays/raster bands are needed for fuzzy overlay.")
    if np.nanmin(data) < 0 or np.nanmax(data) > 1:
        raise InvalidParameterValueException("All data must be in range [0, 1].")
    return data


@beartype
def and_overlay(data: Union[Sequence[np.ndarray], np.ndarray]) -> np.ndarray:
    """Compute an 'and' overlay operation with fuzzy logic.

    Args:
        data: The input data as a series of 2D/3D Numpy arrays or as a 3D Numpy array.
            All found 2D arrays are overlayed. Input data should contain at least 2D Numpy
            arrays and data should be in the range [0, 1].

    Returns:
        2D Numpy array with the result of the 'and' overlay operation. Values are in range [0, 1].

    Raises:
        InvalidDatasetException: If input data contains less than two 2D Numpy arrays/raster bands.
        InvalidParameterValueException: If data values are not in range [0, 1].

    """
    data = _prepare_data_for_fuzzy_overlay(data)
    return data.min(axis=0)


@beartype
def or_overlay(data: Union[Sequence[np.ndarray], np.ndarray]) -> np.ndarray:
    """Compute an 'or' overlay operation with fuzzy logic.

    Args:
        data: The input data as a series of 2D/3D Numpy arrays or as a 3D Numpy array.
            All found 2D arrays are overlayed. Input data should contain at least 2D Numpy
            arrays and data should be in the range [0, 1].

    Returns:
        2D Numpy array with the result of the 'or' overlay operation. Values are in range [0, 1].

    Raises:
        InvalidDatasetException: If input data contains less than two 2D Numpy arrays/raster bands.
        InvalidParameterValueException: If data values are not in range [0, 1].
    """
    data = _prepare_data_for_fuzzy_overlay(data)
    return data.max(axis=0)


@beartype
def product_overlay(data: Union[Sequence[np.ndarray], np.ndarray]) -> np.ndarray:
    """Compute a 'product' overlay operation with fuzzy logic.

    Args:
        data: The input data as a series of 2D/3D Numpy arrays or as a 3D Numpy array.
            All found 2D arrays are overlayed. Input data should contain at least 2D Numpy
            arrays and data should be in the range [0, 1].

    Returns:
        2D Numpy array with the result of the 'product' overlay operation. Values are in range [0, 1].

    Raises:
        InvalidDatasetException: If input data contains less than two 2D Numpy arrays/raster bands.
        InvalidParameterValueException: If data values are not in range [0, 1].
    """
    data = _prepare_data_for_fuzzy_overlay(data)
    return np.prod(data, axis=0)


@beartype
def sum_overlay(data: Union[Sequence[np.ndarray], np.ndarray]) -> np.ndarray:
    """Compute a 'sum' overlay operation with fuzzy logic.

    Args:
        data: The input data as a series of 2D/3D Numpy arrays or as a 3D Numpy array.
            All found 2D arrays are overlayed. Input data should contain at least 2D Numpy
            arrays and data should be in the range [0, 1].

    Returns:
        2D Numpy array with the result of the 'sum' overlay operation. Values are in range [0, 1].

    Raises:
        InvalidDatasetException: If input data contains less than two 2D Numpy arrays/raster bands.
        InvalidParameterValueException: If data values are not in range [0, 1].
    """
    data = _prepare_data_for_fuzzy_overlay(data)
    product_term = np.prod(1 - data, axis=0)
    fuzzy_sum = 1 - product_term
    return fuzzy_sum


@beartype
def gamma_overlay(data: Union[Sequence[np.ndarray], np.ndarray], gamma: float = 0.5) -> np.ndarray:
    """Compute a 'gamma' overlay operation with fuzzy logic.

    Args:
        data: The input data as a series of 2D/3D Numpy arrays or as a 3D Numpy array.
            All found 2D arrays are overlayed. Input data should contain at least 2D Numpy
            arrays and data should be in the range [0, 1].
        gamma: The gamma parameter. With gamma value of 0, the result will be the same as 'product' overlay.
            When gamma is closer to 1, the weight of the 'sum' overlay is increased. Defaults to 0.5.
            Value must be in the range [0, 1].

    Returns:
        2D Numpy array with the result of the 'gamma' overlay operation. Values are in range [0, 1].

    Raises:
        InvalidDatasetException: If input data contains less than two 2D Numpy arrays/raster bands.
        InvalidParameterValueException: If data values or gamma are not in range [0, 1].
    """
    data = _prepare_data_for_fuzzy_overlay(data)
    if gamma < 0 or gamma > 1:
        raise InvalidParameterValueException("The gamma parameter must be in range [0, 1]")

    fuzzy_product = np.prod(data, axis=0)
    product_term_for_sum = np.prod(1 - data, axis=0)
    fuzzy_sum = 1 - product_term_for_sum
    return fuzzy_product ** (1 - gamma) * fuzzy_sum**gamma
