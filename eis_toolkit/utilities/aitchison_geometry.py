from numbers import Number

import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Optional


@beartype
def _normalize(row: pd.Series, sum_value: Number = 1.0) -> pd.Series:
    """
    Normalize the series to a given value.

    If no value is provided, normalize to 1.

    Args:
        row: The series to normalize.

    Returns:
        A series containing the normalized values.
    """
    scale = np.float64(np.sum(row)) / sum_value
    return np.divide(row, scale)


@beartype
def _closure(df: pd.DataFrame, scale: Optional[Number] = None) -> pd.DataFrame:
    """
    Perform the closure operation on the dataframe.

    Assumes the standard simplex, in which the sum of the components of each composition vector is 1.

    Args:
        df: A dataframe of shape (N, D) compositional data.

    Returns:
        A new dataframe of shape (N, D) where each row has been normalized to 1.
    """

    dfc = df.copy()

    for idx, row in df.iterrows():
        dfc.iloc[idx] = _normalize(row, scale) if scale is not None else _normalize(row)

    return dfc
