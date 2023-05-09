from numbers import Number
from typing import Optional, Union

import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Sequence

from eis_toolkit.checks.dataframe import check_columns_valid
from eis_toolkit.exceptions import InvalidColumnException


@beartype
def replace_values(
    data: np.ndarray, values_to_replace: Union[Number, Sequence[Number]], replace_value: Number
) -> np.ndarray:
    """
    Replace one or many values in a Numpy array with a new value. Returns a copy of the input array.

    Args:
        data: Input data as a numpy array.
        values_to_replace: Values to be replaced with the specified replace value.
        replace_value: Value that will replace the specified old values.

    Returns:
        Raster data with replaced values.
    """
    out_data = data.copy()
    return np.where(np.isin(out_data, values_to_replace), replace_value, out_data)


@beartype
def replace_values_df(
    df: pd.DataFrame,
    values_to_replace: Union[Number, Sequence[Number]],
    replace_value: Number,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Replace one or many values in a DataFrame with a new value. Returns a copy of the input array.

    Args:
        df: Input data as a DataFrame.
        values_to_replace: Values to be replaced with the specified replace value.
        replace_value: Value that will replace the specified old values.
        columns: Column names to target the replacement. Defaults to None (all columns).

    Returns:
        DataFrame with replaced values.
    """
    if columns is None:
        columns = df.columns
    elif not check_columns_valid(df, columns):
        raise InvalidColumnException("All selected columns were not found in the input DataFrame.")

    out_df = df.copy()
    for col in columns:
        out_df[col] = out_df[col].replace(values_to_replace, replace_value)

    return out_df
