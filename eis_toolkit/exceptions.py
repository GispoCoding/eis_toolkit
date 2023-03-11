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


class InvalidWindowSizeException(Exception):
    """Exception error class for invalid window size values."""

    pass


class InvalidPixelSizeException(Exception):
    """Exception error class for invalid pixel size."""

    pass


class CoordinatesOutOfBoundsException(Exception):
    """Exception error class for out of bound coordinates."""

    pass


class InvalidWktFormatException(Exception):
    """Exception error for invalid WKT format."""

    pass


class InvalidColumnIndexException(Exception):
    """Exception error for index out of range."""

    pass

class UnFavorableClassDoesntExistException(Exception):
    """Exception error class for failure to generalize classes using the given studentised contrast threshold value. Class 1 (unfavorable class)  doesn't exist"""
  
    pass

class FavorableClassDoesntExistException(Exception):
    """Exception error class for failure to generalize classes using the given studentised contrast threshold value. Class 2 (favorable class)  doesn't exist"""
  
    pass