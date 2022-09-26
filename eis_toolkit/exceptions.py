class NonMatchingCrsException(Exception):
    """Exception error class for crs mismatches."""
    pass


class NotApplicableGeometryTypeException(Exception):
    """Exception error class for not suitable geometry types."""
    pass


class NegativeResamplingFactorException(Exception):
    """Exception error class for negative resampling factors."""

    pass


class InvalidParameterValueException(Exception):
    """Exception error class for invalid parameter values."""

    pass
