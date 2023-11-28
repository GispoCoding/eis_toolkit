import numpy as np
import pandas as pd
from beartype import beartype

from eis_toolkit.exceptions import InvalidColumnException, InvalidParameterValueException
from eis_toolkit.utilities.checks.dataframe import check_dataframe_contains_zeros


@beartype
def _single_pairwise_logratio(x1: float, x2: float) -> np.float64:

    return np.log(x1 / x2)


@beartype
def single_pairwise_logratio(x1: float, x2: float) -> np.float64:
    """
    Perform a pairwise logratio transformation on the given values.

    Args:
        x1: The numerator in the ratio.
        x2: The denominator in the ratio.

    Returns:
        The transformed value.

    Raises:
        InvalidParameterValueException: One or both input values are zero.
    """
    if x1 == 0 or x2 == 0:
        raise InvalidParameterValueException("Input values cannot be zero.")

    return _single_pairwise_logratio(x1, x2)


@beartype
def _pairwise_logratio(df: pd.DataFrame, c1: str, c2: str) -> pd.Series:
    dfc = df.copy()
    dfc[[c1, c2]] = dfc[[c1, c2]].astype(float)

    result = pd.Series([0.0] * df.shape[0])

    for idx, row in dfc.iterrows():
        result[idx] = single_pairwise_logratio(row[c1], row[c2])

    return result


@beartype
def pairwise_logratio(df: pd.DataFrame, c1: str, c2: str) -> pd.Series:
    """
    Perform a pairwise logratio transformation on the given columns.

    Args:
        df: The dataframe containing the columns to use in the transformation.

    Returns:
        A series containing the transformed values.

    Raises:
        InvalidColumnException: One or both of the input columns are not found in the dataframe.
        InvalidParameterValueException: The input columns contain at least one zero value.
    """
    if c1 not in df.columns or c2 not in df.columns:
        raise InvalidColumnException("At least one input column is not found in the dataframe.")

    if check_dataframe_contains_zeros(df[[c1, c2]]):
        raise InvalidParameterValueException("The input columns contain at least one zero value.")

    return _pairwise_logratio(df, c1, c2)
