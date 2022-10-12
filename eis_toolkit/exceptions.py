class NonMatchingCrsException(Exception):
    """Exception error class for crs mismatches."""

    pass


class NotApplicableGeometryTypeException(Exception):
    """Exception error class for not suitable geometry types."""

    pass


class MatchingCrsException(Exception):
    """Exception error class for crs matches."""


class NumericValueSignException(Exception):
    """Exception error class for numeric value sign exception."""

    pass


class InvalidParameterValueException(Exception):
    """Exception error class for invalid parameter values."""

    pass


class NonSquarePixelSizeException(Exception):
    """Exception error class for non-square pixel size."""

    pass


class OutOfBoundsException(Exception):
    """Exception error class for point being out of bounds."""

    pass


class InvalidPixelSizeException(Exception):
    """Exception error class for invalid pixel size."""
