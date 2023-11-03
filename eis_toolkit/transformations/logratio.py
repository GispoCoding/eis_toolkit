import numpy as np
import pandas as pd
from beartype import beartype

from eis_toolkit.exceptions import NonNumericDataException


@beartype
def _all_numeric(df: pd.DataFrame) -> bool:
    return all([t.kind in "iuf" for t in df.dtypes])


@beartype
def _replace_zeros_with_nan_inplace(df: pd.DataFrame, rtol: float = 1e-5, atol: float = 1e-8) -> None:
    if not _all_numeric(df):
        raise NonNumericDataException
    df.loc[:, :] = np.where(np.isclose(df.values, 0.0, rtol=1e-5, atol=1e-8), np.nan, df.values)


@beartype
def _get_rows_with_no_missing_values(df: pd.DataFramem) -> pd.Series:
    return ~df.isna().any(axis=1)


@beartype
def _linear_normalization(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()
    for idx, row in dfc[_get_rows_with_no_missing_values(dfc)].iterrows():
        min = row.iloc[row.argmin()] * 1.0
        max = row.iloc[row.argmax()]
        dfc.iloc[idx] = row.transform(lambda x: (x - min) / (max - min))
    return dfc


@beartype
def _ALR_transform(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame()
