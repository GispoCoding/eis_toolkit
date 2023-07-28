class CoordinatesOutOfBoundsException(Exception):
    """Exception error class for out of bound coordinates."""


class EmptyDataFrameException(Exception):
    """Exception error class raised if the dataframe is empty."""


class FileReadError(Exception):
    """Exception error class for unsupported file exceptions."""


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