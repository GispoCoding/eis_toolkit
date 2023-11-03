from typing import Sequence

import numpy as np
import pandas as pd
from beartype import beartype

from eis_toolkit.exceptions import DataframeShapeException, NonNumericDataException


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
def _linear_normalization(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()
    for idx, row in dfc[_get_rows_with_no_missing_values(dfc)].iterrows():
        min = row.iloc[row.argmin()] * 1.0
        max = row.iloc[row.argmax()]
        dfc.iloc[idx] = row.transform(lambda x: (x - min) / (max - min))
    return dfc


# @beartype
def _ALR_transform(df: pd.DataFrame, columns: Sequence[any], idx=-1) -> pd.DataFrame:
    """
    Perform additive logratio transformation on the selected columns.

    Args:
        df: A Dataframe of shape containing compositional data.
        columns: Columns selected for the transformation.
        ind: The index of the column in the dataframe to be used as denominator.
                If left blank, the last column will be used.

    Returns:
        DataFrame: A new DataFrame of shape (N, D-1) with the ALR transformed values.
    """
    cols = df.columns
    if len(cols) < 2 or len(columns) > len(cols):
        raise DataframeShapeException
    if idx >= len(cols) or idx < -len(cols):
        raise IndexError

    # (todo: check columns exist)

    dfc = df.copy()

    # Only include the relevant columns
    numerators = dfc.loc[:, columns]
    denominator = dfc.iloc[:, idx]

    # dfc = dfc.loc[:, columns].join(dfc.iloc[:, idx])
    ratio = numerators.divide(denominator.iloc[0], axis="index")

    return np.log(ratio)


@beartype
def _ALR_transform_row(row: pd.Series, ind=-1):

    return pd.Series()
