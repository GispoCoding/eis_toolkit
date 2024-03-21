class CoordinatesOutOfBoundsException(Exception):
    """Exception error class for out of bound coordinates."""


class EmptyDataFrameException(Exception):
    """Exception error class raised if the dataframe is empty."""


class FileReadError(Exception):
    """Exception error class for unsupported file exceptions."""


class InconsistentDataTypesException(Exception):
    """Exception error class for inconsistent data types."""


class InvalidArgumentTypeException(Exception):
    """Exception error for invalid argument type."""


class InvalidColumnException(Exception):
    """Exception error for invalid column."""


class InvalidColumnIndexException(Exception):
    """Exception error for invalid column index."""


class InvalidParameterValueException(Exception):
    """Exception error class for invalid parameter values."""


class InvalidPixelSizeException(Exception):
    """Exception error class for invalid pixel size."""


class InvalidRasterBandException(Exception):
    """Expection error class for invalid raster band selection."""


class InvalidWindowSizeException(Exception):
    """Exception error class for invalid window size values."""


class InvalidWktFormatException(Exception):
    """Exception error for invalid WKT format."""


class MatchingCrsException(Exception):
    """Exception error class for CRS matches."""


class MatchingRasterGridException(Exception):
    """Exception error class for raster grid matches."""


class NotApplicableGeometryTypeException(Exception):
    """Exception error class for not suitable geometry types."""


class NonMatchingCrsException(Exception):
    """Exception error class for CRS mismatches."""


class NonMatchingParameterLengthsException(Exception):
    """Exception error class for parameters with different lenghts."""


class NonSquarePixelSizeException(Exception):
    """Exception error class for non-square pixel size."""


class NumericValueSignException(Exception):
    """Exception error class for numeric value sign exception."""


class InvalidNumberOfConv2DLayer(Exception):
    """Exception throws when when the list of conv layers is <= 0."""


class InvalidInputDataException(Exception):
    """Exception throws when the input data is None or invalid."""
