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


class InvalidDataFrameException(Exception):
    """Exception error class thrown if the dtatafram has no rows to analyse."""

    pass


class InvalidNumberOfPrincipalComponent(Exception):
    """Exception error class thrown if the number of principal components should be >= 2."""

    pass


class InvalidInputShapeException(Exception):
    """X train array need to have row and cols."""

    pass


class CanNotMakeCategoricalLabelException(Exception):
    """Exception error class thrown if the labels are not converted to categorical."""

    pass


class InvalidCrossValidationSelected:
    """The selected cv is not implemented."""

    pass


class NumberOfSplitException:
    """Number of cv split should be always > 1."""

    pass
