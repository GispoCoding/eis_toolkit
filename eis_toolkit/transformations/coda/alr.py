import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Optional, Sequence

from eis_toolkit.exceptions import (
    InvalidColumnException,
    InvalidColumnIndexException,
    InvalidCompositionException,
    InvalidParameterValueException,
    NonNumericDataException,
)
from eis_toolkit.utilities.checks.dataframe import (
    check_column_index_in_dataframe,
    check_columns_valid,
    check_dataframe_contains_zeros,
)


@beartype
def _all_numeric(df: pd.DataFrame) -> bool:
    return all([t.kind in "iuf" for t in df.dtypes])


@beartype
def _replace_zeros_with_nan_inplace(df: pd.DataFrame, rtol: float = 1e-5, atol: float = 1e-8) -> None:
    if not _all_numeric(df):
        raise NonNumericDataException
    df.loc[:, :] = np.where(np.isclose(df.values, 0.0, rtol=1e-5, atol=1e-8), np.nan, df.values)


@beartype
def _get_rows_with_no_missing_values(df: pd.DataFrame) -> pd.Series:
    return ~df.isna().any(axis=1)


@beartype
def _alr_transform(df: pd.DataFrame, columns: Sequence[str], denominator_column: str) -> pd.DataFrame:

    ratios = df[columns].div(df[denominator_column], axis=0)
    return np.log(ratios)


@beartype
def alr_transform(
    df: pd.DataFrame, columns: Optional[Sequence[str]] = None, idx: int = -1, keep_redundant_column: bool = False
) -> pd.DataFrame:
    """
    Perform an additive logratio transformation on the selected columns.

    Args:
        df: A dataframe of compositional data.
        columns: Names of the columns to be transformed. If none are given, all columns are used.
        idx: The integer position based index of the column of the dataframe to be used as denominator.
            If not provided, the last column will be used.
        keep_redundant_column: Whether to include the denominator column in the result. If True, it is
            included in the output dataframe regardless of whether it was in the list of given input columns.

    Returns:
        A new dataframe containing the ALR transformed values.

    Raises:
        InvalidColumnException: One or more input columns are not found in the given dataframe, or the
            numerator or denominator columns contain zeros.
        InvalidParameterValueException: Too few columns to perform the transformation.
        InvalidColumnIndexException: The input index for the denominator column is out of bounds.
    """
    if df.isnull().values.any():
        raise InvalidCompositionException("Data contains NaN values.")

    if columns is not None:
        if not check_columns_valid(df, columns):
            raise InvalidColumnException("Not all of the given columns were found in the input DataFrame.")

    columns = [col for col in df.columns] if columns is None else columns

    if len(columns) < 2:
        raise InvalidParameterValueException("Too few columns to perform the transformation.")

    if not check_column_index_in_dataframe(df, idx):
        raise InvalidColumnIndexException("Denominator column index out of bounds.")

    denominator_column = df.columns[idx]

    # TODO: decide: should redundant column maintain its relative
    # position to the other columns in the resulting dataframe?
    if denominator_column not in columns:
        columns.append(denominator_column)

    # TODO: possibly only check the subcomposition columns
    if check_dataframe_contains_zeros(df[columns]):
        raise InvalidColumnException("The given columns or the divisor column contain zeros.")

    if not keep_redundant_column and denominator_column in columns:
        columns.remove(denominator_column)

    return _alr_transform(df, columns, denominator_column)


def inverse_alr():
    """Perform the inverse transformation for a set of ALR transformed data."""
    raise NotImplementedError()
