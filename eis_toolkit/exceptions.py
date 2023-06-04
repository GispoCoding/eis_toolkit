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


class EmptyDataFrameException(Exception):
    """Exception error class thrown if the dataframe is empty."""

    pass


class InvalidWktFormatException(Exception):
    """Exception error for invalid WKT format."""

    pass


class InvalidNumberOfPrincipalComponents(Exception):
    """Exception error class thrown if the number of principal components is less than 2."""

    pass


class InvalidColumnIndexException(Exception):
    """Exception error for index out of range."""

    pass


class InvalidColumnException(Exception):
    """Exception error for invalid column name."""

    pass
