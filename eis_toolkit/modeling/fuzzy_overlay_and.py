import numpy
from eis_toolkit.exceptions import InvalidParameterValueException

def _fuzzy_and( # type: ignore[no-any-unimported]
    indata: numpy.ndarray):
    fuzzyand = indata.max(axis=1)
    return fuzzyand

def fuzzy_and( # type: ignore[no-any-unimported]
    indata: numpy.ndarray):
    """Computes fuzzy union (And)

    Args:
        indata (numpy.ndarray): Array with variables on columns and data points on rows.
        Nodata should be given in which form?

    Returns:
        fuzzyand (numpy.ndarray): One column array giving the fuzzy union for each data point.

    Raises:
        InvalidParameterValueException: Input is not in range [0,1].
    """
    if indata.min() < 0 or indata.max() > 1:
        raise InvalidParameterValueException("Input for fuzzy overlay must be in range [0,1]")

    fuzzyand = _fuzzy_and(indata)
    return fuzzyand