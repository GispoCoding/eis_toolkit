import pandas as pd
from beartype import beartype

from eis_toolkit.checks.dataframe import check_dataframe_contains_nonzero_numbers
from eis_toolkit.exceptions import InvalidColumnException


@beartype
def _CLR_transform(df: pd.DataFrame) -> pd.DataFrame:
    """TODO: docstring."""

    if check_dataframe_contains_nonzero_numbers(df):
        raise InvalidColumnException("The dataframe contains one or more zeros.")

    return pd.DataFrame()
