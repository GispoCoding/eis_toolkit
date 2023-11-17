import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Optional, Sequence, Tuple
from scipy.stats import gmean

from eis_toolkit.exceptions import InvalidColumnException
from eis_toolkit.utilities.aitchison_geometry import _closure
from eis_toolkit.utilities.checks.dataframe import check_columns_valid, check_dataframe_contains_zeros


@beartype
def _centered_ratio(row: pd.Series) -> pd.Series:

    return row / gmean(row)


@beartype
def _CLR_transform(df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:

    columns = df.columns if columns is None else columns

    dfc = df.copy()
    dfc = dfc[columns].apply(_centered_ratio, axis=1)

    return np.log(dfc)


@beartype
def CLR_transform(df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Perform a centered logratio transformation on the selected columns.

    Args:
        df: A dataframe of compositional data.
        columns: Names of the columns to be transformed. If None, all columns are used.

    Returns:
        A new dataframe containing the CLR transformed columns.

    Raises:
        InvalidColumnException: One or more input columns are not found in the given dataframe, or the
            numerator or denominator columns contain zeros.
    """
    # TODO: deal with potential negative values

    if columns is not None:
        if not check_columns_valid(df, columns):
            raise InvalidColumnException("Not all of the given columns were found in the input DataFrame.")

    # TODO: possibly only check the subcomposition columns
    if check_dataframe_contains_zeros(df):
        raise InvalidColumnException("The dataframe contains one or more zeros.")

    return _CLR_transform(df, columns)


def inverse_CLR(df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> Tuple[pd.DataFrame, np.float64]:
    """Perform the inverse transformation for a set of CLR transformed data."""

    return _closure(np.exp(df))
