import numpy

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

    if  gam < 0 or gam > 1:
        raise InvalidParameterValueException("The gamma parameter must be in range [0,1]")

    fuzzygam = _fuzzy_gamma(indata,gam)
    return fuzzygam