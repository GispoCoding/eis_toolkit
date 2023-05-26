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
    columns_numeric = df.columns.select_dtypes(include="number").columns.to_list()
    return all(column in columns_numeric for column in columns)
