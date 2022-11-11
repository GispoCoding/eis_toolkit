import numpy
from eis_toolkit.exceptions import InvalidParameterValueException

def _fuzzy_sum( # type: ignore[no-any-unimported]
    indata: numpy.ndarray):
    subtraction = numpy.subtract(1, indata)
    product = numpy.prod(subtraction,axis=1)
    fuzzysum = numpy.subtract(1, product)
    return fuzzysum

def fuzzy_sum( # type: ignore[no-any-unimported]
    indata: numpy.ndarray):
    """Computes fuzzy Sum (reference to ESRI ArcGIS)

    Args:
        indata (numpy.ndarray): Array with variables on columns and data points on rows.
        Nodata should be given in which form?

    Returns:
        fuzzysum (numpy.ndarray): One column array giving the fuzzy sum for each data point.

    Raises:
        InvalidParameterValueException: Input is not in range [0,1].

    """
    if indata.min() < 0 or indata.max() > 1:
        raise InvalidParameterValueException("Input for fuzzy overlay must be in range [0,1]")

    fuzzysum = _fuzzy_sum(indata)
    return fuzzysum