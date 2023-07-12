from numbers import Number
from typing import Optional, Union

import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Sequence

from eis_toolkit.checks.dataframe import check_columns_valid
from eis_toolkit.checks.parameter import check_dtype_for_int
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
    return np.where(np.isin(out_data, values_to_replace), replace_value, out_data)  # type: ignore


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


@beartype
def cast_scalar_to_int(scalar: Number) -> Number:
    """
    Casts a numerical value to integer type if possible.
    Args:
        scalar: Input scalar value.
    Returns:
        The input scalar as an integer if it can be cast, else the original scalar.
    """
    if check_dtype_for_int(scalar) == True:
        return int(scalar)
    else:
        return scalar


def get_min_int_type(data: np.ndarray | Number) -> np.dtype:
    if isinstance(data, np.ndarray):
        data_min = np.min(data)
        data_max = np.max(data)

        if np.iinfo(np.int8).min <= data_min <= data_max <= np.iinfo(np.int8).max:
            return np.int8
        elif np.iinfo(np.uint8).min <= data_min <= data_max <= np.iinfo(np.uint8).max:
            return np.uint8
        elif np.iinfo(np.int16).min <= data_min <= data_max <= np.iinfo(np.int16).max:
            return np.int16
        elif np.iinfo(np.uint16).min <= data_min <= data_max <= np.iinfo(np.uint16).max:
            return np.uint16
        elif np.iinfo(np.int32).min <= data_min <= data_max <= np.iinfo(np.int32).max:
            return np.int32
        elif np.iinfo(np.uint32).min <= data_min <= data_max <= np.iinfo(np.uint32).max:
            return np.uint32
        elif np.iinfo(np.int64).min <= data_min <= data_max <= np.iinfo(np.int64).max:
            return np.int64
        elif np.iinfo(np.uint64).min <= data_min <= data_max <= np.iinfo(np.uint64).max:
            return np.uint64

    if isinstance(data, Number):
        data = cast_scalar_to_int(data)

        if isinstance(data, int):
            if np.iinfo(np.int8).min <= data <= np.iinfo(np.int8).max:
                return np.int8
            elif np.iinfo(np.uint8).min <= data <= np.iinfo(np.uint8).max:
                return np.uint8
            elif np.iinfo(np.int16).min <= data <= np.iinfo(np.int16).max:
                return np.int16
            elif np.iinfo(np.uint16).min <= data <= np.iinfo(np.uint16).max:
                return np.uint16
            elif np.iinfo(np.int32).min <= data <= np.iinfo(np.int32).max:
                return np.int32
            elif np.iinfo(np.uint32).min <= data <= np.iinfo(np.uint32).max:
                return np.uint32
            elif np.iinfo(np.int64).min <= data <= np.iinfo(np.int64).max:
                return np.int64
            elif np.iinfo(np.uint64).min <= data <= np.iinfo(np.uint64).max:
                return np.uint64
        else:
            return np.min_scalar_type(data)
