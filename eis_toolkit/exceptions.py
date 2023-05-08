class NonMatchingCrsException(Exception):
    """Exception error class for crs mismatches."""


class NotApplicableGeometryTypeException(Exception):
    """Exception error class for not suitable geometry types."""


class MatchingCrsException(Exception):
    """Exception error class for crs matches."""


class NumericValueSignException(Exception):
    """Exception error class for numeric value sign exception."""


class InvalidParameterValueException(Exception):
    """Exception error class for invalid parameter values."""


class NonSquarePixelSizeException(Exception):
    """Exception error class for non-square pixel size."""


class InvalidWindowSizeException(Exception):
    """Exception error class for invalid window size values."""


class InvalidPixelSizeException(Exception):
    """Exception error class for invalid pixel size."""


class CoordinatesOutOfBoundsException(Exception):
    """Exception error class for out of bound coordinates."""


class EmptyDataFrameException(Exception):
    """Exception error class thrown if the dataframe is empty."""


class InvalidWktFormatException(Exception):
    """Exception error for invalid WKT format."""


class InvalidNumberOfPrincipalComponents(Exception):
    """Exception error class thrown if the number of principal components is less than 2."""


class InvalidColumnException(Exception):
    """Exception error for invalid column."""


class InvalidColumnIndexException(Exception):
    """Exception error for invalid column index."""


class InvalidArgumentTypeException(Exception):
    """Exception error for invalid argument type."""


class InvalidRasterBandException(Exception):
    """Expection error class for invalid raster band selection."""


class FileReadError(Exception):
    """Exception error class for unsupported file exceptions."""
