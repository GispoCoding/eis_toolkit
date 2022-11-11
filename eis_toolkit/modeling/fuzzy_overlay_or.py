import numpy
from eis_toolkit.exceptions import InvalidParameterValueException

def _fuzzy_or( # type: ignore[no-any-unimported]
    indata: numpy.ndarray):
    fuzzyor = indata.min(axis=1)
    return fuzzyor

def fuzzy_or( # type: ignore[no-any-unimported]
    indata: numpy.ndarray):
    """Computes fuzzy intersection (Or)

    Args:
        indata (numpy.ndarray): Array with variables on columns and data points on rows.
        Nodata should be given in which form?

    Returns:
        fuzzyor (numpy.ndarray): One column array giving the fuzzy intersection for each data point.

    Raises:
        InvalidParameterValueException: Input is not in range [0,1].
    """
    if indata.min() < 0 or indata.max() > 1:
        raise InvalidParameterValueException("Input for fuzzy overlay must be in range [0,1]")

    fuzzyor = _fuzzy_or(indata)
    return fuzzyor