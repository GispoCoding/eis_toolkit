class CoordinatesOutOfBoundsException(Exception):
    """Exception error class for out of bound coordinates."""


class EmptyDataFrameException(Exception):
    """Exception error class raised if the dataframe is empty."""


class FileReadError(Exception):
    """Exception error class for unsupported file exceptions."""


class FileWriteError(Exception):
    """Exception error class for writing file exceptions."""


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


class InvalidWktFormatException(Exception):
    """Exception error for invalid WKT format."""


class MatchingCrsException(Exception):
    """Exception error class for CRS matches."""


class MatchingRasterGridException(Exception):
    """Exception error class for raster grid matches."""


class NonMatchingCrsException(Exception):
    """Exception error class for CRS mismatches."""


class NonMatchinParameterLengthsException(Exception):
    """Exception error class for parameters with different lenghts."""


class NotApplicableGeometryTypeException(Exception):
    """Exception error class for not suitable geometry types."""


class NumericValueSignException(Exception):
    """Exception error class for numeric value sign exception."""
