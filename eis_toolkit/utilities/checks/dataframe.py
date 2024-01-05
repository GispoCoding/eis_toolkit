import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Sequence


@beartype
def check_columns_valid(df: pd.DataFrame, columns: Sequence[str]) -> bool:
    """
    Check that all specified columns are in the dataframe.

    Args:
        df: Dataframe to be checked.
        columns: Column names.

    Returns:
        True if all columns are found in the dataframe, otherwise False.
    """
    return all(column in df.columns for column in columns)


@beartype
def check_columns_numeric(df: pd.DataFrame, columns: Sequence[str]) -> bool:
    """
    Check that all specified columns are numeric.

    Args:
        df: Dataframe to be checked.
        columns: Column names.

    Returns:
        True if all columns are numeric, otherwise False.
    """
    columns_numeric = df[columns].select_dtypes(include="number").columns.to_list()
    return all(column in columns_numeric for column in columns)


@beartype
def check_columns_categorical(df: pd.DataFrame, columns: Sequence[str], max_unique_values: int = 20) -> bool:
    """
    Check that all specified columns are categorical.

    Args:
        df: Dataframe to be checked.
        columns: Column names.
        max_unique_values: Maximum number of unique values for numeric columns to be considered categorical.
            Defaults to 20.

    Returns
        True if all columns are categorical, otherwise False.
    """
    numeric_dtypes = ["int64", "float64"]  # Expand this list?

    return all(
        (
            isinstance(df[column].dtype, pd.CategoricalDtype)
            or (df[column].dtype == bool)
            or (df[column].dtype.name in numeric_dtypes and df[column].nunique() <= max_unique_values)
        )
        for column in columns
    )


def check_empty_dataframe(df: pd.DataFrame) -> bool:
    """Check if the dataframe is empty.

    Args:
        df: Dataframe to be checked.

    Return:
        True if dataframe is empty, otherwise False.
    """
    return df.empty


@beartype
def check_dataframe_contains_zeros(df: pd.DataFrame) -> bool:
    """
    Check if the dataframe contains any zeros.

    Args:
        df: Dataframe to be checked.
    """
    return 0 in df.values


@beartype
def check_dataframe_contains_only_positive_numbers(df: pd.DataFrame) -> np.bool_:
    """
    Check that the dataframe only contains positive, nonzero values.

    Args:
        df: Dataframe to be checked.
    """
    return np.all([val > 0 for val in df.values])
