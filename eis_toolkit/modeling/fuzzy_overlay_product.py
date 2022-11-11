import numpy
from eis_toolkit.exceptions import InvalidParameterValueException

def _fuzzy_prod( # type: ignore[no-any-unimported]
    indata: numpy.ndarray):
    fuzzyprod = numpy.prod(indata,axis=1)
    return fuzzyprod

def fuzzy_prod( # type: ignore[no-any-unimported]
    indata: numpy.ndarray):
    """Computes fuzzy product (reference to ESRI ArcGIS)

    Args:
        indata (numpy.ndarray): Array with variables on columns and data points on rows.
        Nodata should be given in which form?

    Returns:
        fuzzyprod (numpy.ndarray): One column array giving the fuzzy product for each data point.

    Raises:
        InvalidParameterValueException: Input is not in range [0,1].

    """
    if indata.min() < 0 or indata.max() > 1:
        raise InvalidParameterValueException("Input for fuzzy overlay must be in range [0,1]")

    fuzzyprod = _fuzzy_prod(indata)
    return fuzzyprod