from numbers import Number

import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Optional, Sequence
from scipy.stats import gmean

from eis_toolkit.exceptions import InvalidColumnException, NumericValueSignException
from eis_toolkit.utilities.aitchison_geometry import _closure
from eis_toolkit.utilities.checks.compositional import check_in_simplex_sample_space
from eis_toolkit.utilities.miscellaneous import rename_columns, rename_columns_by_pattern


@beartype
def _centered_ratio(row: pd.Series) -> pd.Series:
    return row / gmean(row)


@beartype
def _clr_transform(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()
    dfc = dfc.apply(_centered_ratio, axis=1)

    return np.log(dfc)


@beartype
def clr_transform(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    scale: Optional[Number] = None,
) -> pd.DataFrame:
    """
    Perform a centered logratio transformation on the data.

    Args:
        df: A dataframe of compositional data.
        columns: The names of the columns to be transformed.
        scale: The value to which each composition should be normalized. Eg., if the composition is expressed
            as percentages, scale=100. Closure is not performed by default.

    Returns:
        A new dataframe containing the CLR transformed data.

    Raises:
        InvalidColumnException: The input column(s) not found in the dataframe.
        InvalidCompositionException: Data is not normalized to the expected value.
        NumericValueSignException: Data contains zeros or negative values.
    """

    if columns:
        invalid_columns = [col for col in columns if col not in df.columns]
        if invalid_columns:
            raise InvalidColumnException(f"The following columns were not found in the dataframe: {invalid_columns}.")
        columns_to_transform = columns
        df = df[columns_to_transform]
    else:
        columns_to_transform = df.columns.to_list()

    if scale is not None:
        df = _closure(df, scale)

    check_in_simplex_sample_space(df)

    return rename_columns_by_pattern(_clr_transform(df))


@beartype
def _inverse_clr(df: pd.DataFrame, scale: Number = 1.0) -> pd.DataFrame:
    return _closure(np.exp(df), scale)


@beartype
def inverse_clr(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    colnames: Optional[Sequence[str]] = None,
    scale: Number = 1.0,
) -> pd.DataFrame:
    """
    Perform the inverse transformation for a set of CLR transformed data.

    Args:
        df: A dataframe of CLR transformed compositional data.
        columns: The names of the columns to be transformed.
        colnames: List of column names to rename the columns to.
        scale: The value to which each composition should be normalized. Eg., if the composition is expressed
            as percentages, scale=100.

    Returns:
        A dataframe containing the inverse transformed data.

    Raises:
        InvalidColumnException: The input column(s) not found in the dataframe.
        NumericValueSignException: The input scale value is zero or less.
    """
    if scale <= 0:
        raise NumericValueSignException("The scale value should be positive.")

    if columns:
        invalid_columns = [col for col in columns if col not in df.columns]
        if invalid_columns:
            raise InvalidColumnException(f"The following columns were not found in the dataframe: {invalid_columns}.")
        columns_to_transform = columns
    else:
        columns_to_transform = df.columns.to_list()

    dfc = df.copy()
    dfc = dfc[columns_to_transform]

    inverse_data = _inverse_clr(dfc, scale)

    if colnames:
        return rename_columns(inverse_data, colnames)

    return inverse_data
