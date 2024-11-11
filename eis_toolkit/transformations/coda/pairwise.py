from numbers import Number

import numpy as np
import pandas as pd
from beartype import beartype

from eis_toolkit.exceptions import InvalidColumnException, InvalidParameterValueException
from eis_toolkit.utilities.checks.dataframe import check_dataframe_contains_zeros


@beartype
def _single_pairwise_logratio(numerator: Number, denominator: Number) -> np.float64:

    return np.log(numerator / float(denominator))


@beartype
def single_pairwise_logratio(numerator: Number, denominator: Number) -> np.float64:
    """
    Perform a pairwise logratio transformation on the given values.

    Args:
        numerator: The numerator in the ratio.
        denominator: The denominator in the ratio.

    Returns:
        The transformed value.

    Raises:
        InvalidParameterValueException: One or both input values are zero.
    """
    if numerator == 0 or denominator == 0:
        raise InvalidParameterValueException("Input values cannot be zero.")

    return _single_pairwise_logratio(numerator, denominator)


@beartype
def _pairwise_logratio(df: pd.DataFrame, numerator_column: str, denominator_column: str) -> pd.Series:
    dfc = df.copy()

    result = pd.Series([0.0] * df.shape[0])

    for idx, row in dfc.iterrows():
        result[idx] = single_pairwise_logratio(row[numerator_column], row[denominator_column])

    return result


@beartype
def pairwise_logratio(df: pd.DataFrame, numerator_column: str, denominator_column: str) -> pd.Series:
    """
    Perform a pairwise logratio transformation on the given columns.

    Args:
        df: The dataframe containing the columns to use in the transformation.
        numerator_column: The name of the column to use as the numerator column.
        denominator_column: The name of the column to use as the denominator.

    Returns:
        A series containing the transformed values.

    Raises:
        InvalidColumnException: One or both of the input columns are not found in the dataframe.
        InvalidParameterValueException: The input columns contain at least one zero value.
    """
    if numerator_column not in df.columns or denominator_column not in df.columns:
        raise InvalidColumnException("At least one input column is not found in the dataframe.")

    if check_dataframe_contains_zeros(df[[numerator_column, denominator_column]]):
        raise InvalidParameterValueException("The input columns contain at least one zero value.")

    return _pairwise_logratio(df, numerator_column, denominator_column)
