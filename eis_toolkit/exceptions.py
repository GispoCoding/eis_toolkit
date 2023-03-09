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


class InvalideContentOfInputDataFrame(Exception):
    """Exception error class for missing or wrong values in DataFrame."""

    pass


class FileReadWriteError(Exception):
    """Exception error class for an error ocurs during read, write or delete a file."""

    pass


class MissingFileOrPath(Exception):
    """Exception error class for an error ocurs during read, write or delete a file."""

    pass


class NonMatchingImagesExtend(Exception):
    """Exception error class for extend mismatches."""

    pass


class ModelIsNotFitted(Exception):
    """Exception error class for not fitted model."""

    pass