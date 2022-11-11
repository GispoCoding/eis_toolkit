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

def _fuzzy_gamma( # type: ignore[no-any-unimported]
    indata: numpy.ndarray, gam: float):
    subtraction = numpy.subtract(1, indata)
    product = numpy.prod(subtraction,axis=1)
    fuzzysum = numpy.subtract(1, product)
    fuzzyprod = numpy.prod(indata,axis=1)
    print(fuzzysum)
    fuzzygam = fuzzyprod**(1-gam)*fuzzysum**gam
    return fuzzygam

def fuzzy_gamma( # type: ignore[no-any-unimported]
    indata: numpy.ndarray, gam: float):
    """Computes fuzzy gamma (reference to ESRI ArcGIS)

    Args:
        indata (numpy.ndarray): Data array with variables on columns and data points on rows.
        gam (float): Gamma parameter
        Nodata should be given in which form?

    Returns:
        fuzzygam (numpy.ndarray): One column array giving the fuzzy gamma value for each data point.

    Raises:
        InvalidParameterValueException: Input is not in range [0,1].

    """
    if indata.min() < 0 or indata.max() > 1:
        raise InvalidParameterValueException("Input for fuzzy overlay must be in range [0,1]")

    if  gam < 0 or gam > 1 or numpy.isnan(gam):
        raise InvalidParameterValueException("The gamma parameter must be in range [0,1]")

    fuzzygam = _fuzzy_gamma(indata,gam)
    return fuzzygam