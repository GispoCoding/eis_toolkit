from typing import Iterable

import pandas as pd


def check_columns_valid(df: pd.DataFrame, columns: Iterable[str]):
    """
    Check that all specified columns are in the dataframe.

    Args:
        df: Dataframe to be checked.
        columns: Column names.

    Returns:
        True if all columns are found in the dataframe, otherwise False.
    """
    check = all(column in df.columns for column in columns)
    return check
