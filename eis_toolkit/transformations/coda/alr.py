from numbers import Number

import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Optional, Sequence

from eis_toolkit.exceptions import InvalidColumnException, NumericValueSignException
from eis_toolkit.utilities.aitchison_geometry import _closure
from eis_toolkit.utilities.checks.compositional import check_in_simplex_sample_space
from eis_toolkit.utilities.miscellaneous import rename_columns_by_pattern


@beartype
def _alr_transform(df: pd.DataFrame, columns: Sequence[str], denominator_column: str) -> pd.DataFrame:

    ratios = df[columns].div(df[denominator_column], axis=0)
    return np.log(ratios)


@beartype
def alr_transform(
    df: pd.DataFrame, column: Optional[str] = None, keep_denominator_column: bool = False
) -> pd.DataFrame:
    """
    Perform an additive logratio transformation on the data.

    Args:
        df: A dataframe of compositional data.
        column: The name of the column to be used as the denominator column.
        keep_denominator_column: Whether to include the denominator column in the result. If True, the returned
            dataframe retains its original shape.

    Returns:
        A new dataframe containing the ALR transformed data.

    Raises:
        InvalidColumnException: The input column isn't found in the dataframe.
        InvalidCompositionException: Data is not normalized to the expected value.
        NumericValueSignException: Data contains zeros or negative values.
    """
    check_in_simplex_sample_space(df)

    if column is not None and column not in df.columns:
        raise InvalidColumnException(f"The column {column} was not found in the dataframe.")

    column = column if column is not None else df.columns[-1]

    columns = [col for col in df.columns]

    if not keep_denominator_column and column in columns:
        columns.remove(column)

    return rename_columns_by_pattern(_alr_transform(df, columns, column))


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
