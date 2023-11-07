import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Optional, Sequence

from eis_toolkit.checks.dataframe import (
    check_column_index_in_dataframe,
    check_columns_valid,
    check_dataframe_contains_nonzero_numbers,
)
from eis_toolkit.exceptions import (
    InvalidColumnException,
    InvalidColumnIndexException,
    InvalidParameterValueException,
    NonNumericDataException,
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
def _ALR_transform(
    df: pd.DataFrame, columns: Optional[Sequence[str]] = None, idx: int = -1, keep_redundant_column: bool = False
) -> pd.DataFrame:
    """
    Perform additive logratio transformation on the selected columns.

    Args:
        df: A dataframe containing compositional data.
        columns: Names of the columns to be transformed. If none are given, all columns are used.
        idx: The integer position based index of the column of the dataframe to be used as denominator.
            If not provided, the last column will be used.
        keep_redundant_column: Whether to include the denominator column in the result. If True, it is
            included in the output dataframe regardless of whether it was in the list of given input columns.

    Returns:
        DataFrame: A new dataframe containing the ALR transformed values.

    Raises:
        InvalidColumnException: One or more input columns are not found in the given dataframe, or the
            numerator or denominator columns contain zeros.
        InvalidParameterValueException: Too few columns to perform the transformation.
        InvalidColumnIndexException: The input index for the denominator column is out of bounds.
    """
    if columns is not None:
        if check_columns_valid(df, columns) is False:
            raise InvalidColumnException("Not all of the given columns were found in the input DataFrame.")

    columns = [col for col in df.columns] if columns is None else columns

    if len(columns) < 2:
        raise InvalidParameterValueException("Too few columns to perform the transformation.")

    if check_column_index_in_dataframe(df, idx) is False:
        raise InvalidColumnIndexException("Denominator column index out of bounds.")

    denominator_column = df.columns[idx]

    if denominator_column not in columns:
        columns.append(denominator_column)

    if check_dataframe_contains_nonzero_numbers(df[columns]):
        raise InvalidColumnException("The given columns or the divisor column contain zeros.")

    if not keep_redundant_column and denominator_column in columns:
        columns.remove(denominator_column)

    ratios = df[columns].div(df[denominator_column], axis=0)
    return np.log(ratios)


def _inverse_ALR(df: pd.DataFrame, denominator_column: pd.Series) -> pd.DataFrame:
    # TODO: implement
    return pd.DataFrame()
