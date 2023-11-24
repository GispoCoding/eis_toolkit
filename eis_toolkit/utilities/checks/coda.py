import functools
from typing import Callable

import numpy as np
import pandas as pd
from beartype import beartype

from eis_toolkit.exceptions import InvalidCompositionException, NumericValueSignException
from eis_toolkit.utilities.checks.dataframe import check_dataframe_contains_only_positive_numbers


def check_compositional(func: Callable) -> Callable:
    """
    Check if the dataframe argument in the callable is a valid composition.

    A wrapper function that can be used as a decorator.

    Args:
        func: The function to wrap.

    Raises:
        InvalidCompositionException: Data is not normalized to the expected value.
        NumericValueSignException: Data contains zeros or negative values.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for arg in args:
            if type(arg) is pd.DataFrame:
                check_in_simplex_sample_space(arg)
        for kwarg in kwargs.values():
            if kwarg == "df":
                check_in_simplex_sample_space(arg)
        return func(*args, **kwargs)

    return wrapper


@beartype
def check_in_simplex_sample_space(df: pd.DataFrame, k: np.float64 = None) -> None:
    """
    Check that the compositions represented by the data rows belong to a simplex sample space.

    Checks that each compositional data point belongs to the set of positive real numbers.
    Checks that each composition is normalized to the same value.

    Args:
        df: The dataframe to check.
        k: The expected sum of each row. If None, simply checks that the sum of each row is equal.

    Returns:
        True if values are valid and the sum of each row is k.

    Raises:
        InvalidCompositionException: Data is not normalized to the expected value.
        NumericValueSignException: Data contains zeros or negative values.
    """
    if df.isnull().values.any():
        raise InvalidCompositionException("Data contains NaN values.")

    if not check_dataframe_contains_only_positive_numbers(df):
        raise NumericValueSignException("Data contains zeros or negative values.")

    df_sum = np.sum(df, axis=1)
    expected_sum = k if k is not None else df_sum.iloc[0]
    if len(df_sum[df_sum.iloc[:] != expected_sum]) != 0:
        raise InvalidCompositionException("Not each composition is normalized to the same value.")

    return None
