from numbers import Number

import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Optional, Sequence
from scipy.stats import gmean

from eis_toolkit.exceptions import NumericValueSignException
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
def clr_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform a centered logratio transformation on the data.

    Args:
        df: A dataframe of compositional data.

    Returns:
        A new dataframe containing the CLR transformed data.

    Raises:
        InvalidCompositionException: Data is not normalized to the expected value.
        NumericValueSignException: Data contains zeros or negative values.
    """
    check_in_simplex_sample_space(df)
    return rename_columns_by_pattern(_clr_transform(df))


@beartype
def _inverse_clr(df: pd.DataFrame, colnames: Optional[Sequence[str]] = None, scale: Number = 1.0) -> pd.DataFrame:
    inverse = _closure(np.exp(df), scale)

    if colnames is not None:
        return rename_columns(inverse, colnames)

    return inverse


@beartype
def inverse_clr(df: pd.DataFrame, colnames: Optional[Sequence[str]] = None, scale: Number = 1.0) -> pd.DataFrame:
    """
    Perform the inverse transformation for a set of CLR transformed data.

    Args:
        df: A dataframe of CLR transformed compositional data.
        colnames: List of column names to rename the columns to.
        scale: The value to which each composition should be normalized. Eg., if the composition is expressed
            as percentages, scale=100.

    Returns:
        A dataframe containing the inverse transformed data.

    Raises:
        NumericValueSignException: The input scale value is zero or less.
    """
    if scale <= 0:
        raise NumericValueSignException("The scale value should be positive.")

    return _inverse_clr(df, colnames, scale)
