class NonMatchingCrsException(Exception):
    """Exception error class for crs mismatches."""

    pass


class NotApplicableGeometryTypeException(Exception):
    """Exception error class for not suitable geometry types."""

    pass


class InvalidWindowSizeException(Exception):
    """Exception error class for invalid window size values."""
    pass


class CoordinatesOutOfBoundExeption(Exception):
    """Exception error class for out of bound coordinates."""
    pass
