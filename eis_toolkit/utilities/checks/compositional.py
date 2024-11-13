from numbers import Number

import pandas as pd
from beartype import beartype

from eis_toolkit.exceptions import InvalidCompositionException, NumericValueSignException
from eis_toolkit.utilities.checks.dataframe import check_dataframe_contains_only_positive_numbers


@beartype
def check_in_simplex_sample_space(df: pd.DataFrame, tolerance: Number = 0.0001) -> None:
    """
    Check that the compositions represented by the data rows belong to a simplex sample space.

    Checks that data has not NaN values.
    Checks that each compositional data point belongs to the set of positive real numbers.
    Checks that input dataframe is closed to either 1 or 100.

    Args:
        df: The dataframe to check.
        tolerance: Small tolerance value to allow floating-point imprecision.

    Returns:
        None.

    Raises:
        InvalidCompositionException: Data is not within the expected simplex sample space.
        NumericValueSignException: Data contains zeros or negative values.
    """
    if df.isnull().values.any():
        raise InvalidCompositionException("Data contains NaN values.")

    if not check_dataframe_contains_only_positive_numbers(df):
        raise NumericValueSignException("Data contains zeros or negative values.")

    row_sums = df.sum(axis=1)
    closed_to_one = (row_sums - 1).abs() < tolerance
    closed_to_hundred = (row_sums - 100).abs() < tolerance

    if not closed_to_one.all() and not closed_to_hundred.all():
        raise InvalidCompositionException(f"Input data is not closed to 1 or 100 within tolerance of {tolerance}.")

    return None
