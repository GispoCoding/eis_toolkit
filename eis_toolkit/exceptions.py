class CoordinatesOutOfBoundsException(Exception):
    """Exception error class for out of bound coordinates."""


class ClassificationFailedException(Exception):
    """Exception error class for classification failures."""


class EmptyDataException(Exception):
    """Exception error class raised if input data is empty."""


class EmptyDataFrameException(Exception):
    """Exception error class raised if the dataframe is empty."""


class FileReadError(Exception):
    """Exception error class for unsupported file exceptions."""


class InconsistentDataTypesException(Exception):
    """Exception error class for inconsistent data types."""


class InvalidColumnException(Exception):
    """Exception error for invalid column."""


class InvalidColumnIndexException(Exception):
    """Exception error for invalid column index."""


class InvalidDataShapeException(Exception):
    """Exception error for datasets with invalid shapes."""


class InvalidModelTypeException(Exception):
    """Exception error for invalid model type."""


class InvalidParameterValueException(Exception):
    """Exception error class for invalid parameter values."""


class InvalidRasterBandException(Exception):
    """Expection error class for invalid raster band selection."""


class InvalidWktFormatException(Exception):
    """Exception error for invalid WKT format."""


class MatchingCrsException(Exception):
    """Exception error class for CRS matches."""


class MatchingRasterGridException(Exception):
    """Exception error class for raster grid matches."""


class GeometryTypeException(Exception):
    """Exception error class for not suitable geometry types."""


class NonMatchingCrsException(Exception):
    """Exception error class for CRS mismatches."""


class NonMatchingRasterMetadataException(Exception):
    """Exception error class for raster metadata mismatches."""


class NonMatchingParameterLengthsException(Exception):
    """Exception error class for parameters with different lenghts."""


class NonSquarePixelSizeException(Exception):
    """Exception error class for non-square pixel size."""


class NumericValueSignException(Exception):
    """Exception error class for numeric value sign exception."""


class InvalidDatasetException(Exception):
    """Exception error class when the dataset is null."""


class NonNumericDataException(Exception):
    """Exception error class for when the given data includes non-numeric values."""


class InvalidCompositionException(Exception):
    """Exception error class for when the data is not in suitable form for compositional data transforms."""


class SampleSizeExceededException(Exception):
    """Exception error class for when the data exceeds maximum sample size."""
