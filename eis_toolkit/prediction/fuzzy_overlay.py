from enum import Enum
from numbers import Number
from typing import Optional

import numpy as np
from beartype import beartype

from eis_toolkit.exceptions import InvalidParameterValueException


class FuzzyMethod(Enum):
    """Enum for fuzzy methods."""

    AND = 1
    OR = 2
    PRODUCT = 3
    SUM = 4
    GAMMA = 5


def _fuzzy_overlay(data: np.ndarray, method: FuzzyMethod, gamma: Optional[float]) -> np.ndarray:
    if method == FuzzyMethod.AND:  # Intersection
        return data.min(axis=0)
    elif method == FuzzyMethod.OR:  # Union
        return data.max(axis=0)
    elif method == FuzzyMethod.PRODUCT:
        return np.prod(data, axis=0)
    elif method == FuzzyMethod.SUM:
        return data.sum(axis=0) - np.prod(data, axis=0)
    elif method == FuzzyMethod.GAMMA:
        fuzzy_sum = data.sum(axis=0) - np.prod(data, axis=0)
        fuzzy_product = np.prod(data, axis=0)
        return fuzzy_product ** (1 - gamma) * fuzzy_sum**gamma


@beartype
def fuzzy_overlay(rasters_data: np.ndarray, method: FuzzyMethod, gamma: Optional[Number] = None) -> np.ndarray:
    """Compute fuzzy overlay using the specified method.

    Args:
        data: The input data as a 3D Numpy array. The 3D array consists of 2D arrays that represent single raster
            bands. Data points must be in the range [0, 1].
        method: The overlay method to use. Options are AND, OR, PROD, SUM and GAMMA.
        gamma: The gamma parameter for the GAMMA method. Must be in the range [0, 1] if provided.

    Returns:
        2D Numpy array with the results of the overlay operation.

    Raises:
        InvalidParameterValueException: If data values or gamma is not in range [0, 1] or gamma is not
            provided when GAMMA overlay method is selected.
    """

    if any(raster.min() < 0 or raster.max() > 1 for raster in rasters_data):
        raise InvalidParameterValueException("All data must be in range [0, 1]")

    if gamma is None and method == FuzzyMethod.GAMMA:
        raise InvalidParameterValueException("The gamma parameter must be provided for the GAMMA method")

    if gamma and (gamma < 0 or gamma > 1):
        raise InvalidParameterValueException("The gamma parameter must be in range [0, 1]")

    return _fuzzy_overlay(rasters_data, method, gamma)
