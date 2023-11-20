import numpy as np
from beartype import beartype

from eis_toolkit.exceptions import InvalidParameterValueException


@beartype
def _single_pairwise_logratio(x1: float, x2: float) -> np.float64:

    return np.log(x1 / x2)


@beartype
def single_pairwise_logratio(x1: float, x2: float) -> np.float64:
    """
    Perform a pairwise logratio transformation on the given values.

    Args:
        x1: The numerator in the ratio.
        x2: The denominator in the ratio.

    Returns:
        The transformed value.

    Raises:
        InvalidParameterValueException: One or both input values are zero.
    """
    if x1 == 0 or x2 == 0:
        raise InvalidParameterValueException("Input values cannot be zero.")

    return _single_pairwise_logratio(x1, x2)
