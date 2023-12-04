from numbers import Number

import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Sequence

from eis_toolkit.exceptions import InvalidColumnIndexException, NumericValueSignException
from eis_toolkit.utilities.aitchison_geometry import _closure
from eis_toolkit.utilities.checks.compositional import check_in_simplex_sample_space
from eis_toolkit.utilities.checks.dataframe import check_column_index_in_dataframe
from eis_toolkit.utilities.miscellaneous import rename_columns_by_pattern


@beartype
def _alr_transform(df: pd.DataFrame, columns: Sequence[str], denominator_column: str) -> pd.DataFrame:

    ratios = df[columns].div(df[denominator_column], axis=0)
    return np.log(ratios)


@beartype
def alr_transform(df: pd.DataFrame, column: int = -1, keep_denominator_column: bool = False) -> pd.DataFrame:
    """
    Perform an additive logratio transformation on the data.

    Args:
        df: A dataframe of compositional data.
        column: The integer position based index of the column of the dataframe to be used as denominator.
            If not provided, the last column will be used.
        keep_denominator_column: Whether to include the denominator column in the result. If True, the returned
            dataframe retains its original shape.

    Returns:
        A new dataframe containing the ALR transformed data.

    Raises:
        InvalidColumnIndexException: The input index for the denominator column is out of bounds.
        InvalidCompositionException: Data is not normalized to the expected value.
        NumericValueSignException: Data contains zeros or negative values.
    """
    check_in_simplex_sample_space(df)

    if not check_column_index_in_dataframe(df, column):
        raise InvalidColumnIndexException("Denominator column index out of bounds.")

    denominator_column = df.columns[column]
    columns = [col for col in df.columns]

    if not keep_denominator_column and denominator_column in columns:
        columns.remove(denominator_column)

    return rename_columns_by_pattern(_alr_transform(df, columns, denominator_column))


@beartype
def _inverse_alr(df: pd.DataFrame, denominator_column: str, scale: Number = 1.0) -> pd.DataFrame:
    dfc = df.copy()

    if denominator_column not in dfc.columns.values:
        # Add the denominator column
        dfc[denominator_column] = 0.0

    return _closure(np.exp(dfc), scale)


@beartype
def inverse_alr(df: pd.DataFrame, denominator_column: str, scale: Number = 1.0) -> pd.DataFrame:
    """
    Perform the inverse transformation for a set of ALR transformed data.

    Args:
        df: A dataframe of ALR transformed compositional data.
        denominator_column: The name of the denominator column.
        scale: The value to which each composition should be normalized. Eg., if the composition is expressed
            as percentages, scale=100.

    Returns:
        A dataframe containing the inverse transformed data.

    Raises:
        NumericValueSignException: The input scale value is zero or less.
    """
    if scale <= 0:
        raise NumericValueSignException("The scale value should be positive.")

    return _inverse_alr(df, denominator_column, scale)
