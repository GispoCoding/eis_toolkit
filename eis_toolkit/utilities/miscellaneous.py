from numbers import Number
from typing import Optional, Union, List, Tuple, Any

import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Iterable

from eis_toolkit.checks.dataframe import check_columns_valid
from eis_toolkit.checks.parameter import check_dtype_for_int
from eis_toolkit.exceptions import InvalidColumnException


def expand_and_zip(selection: List[Any], *args: Union[List[Any], Tuple[Any]], **kwargs: Any) -> List[Tuple[Any]]:
    """
    Expands and zips a selection with additional arguments and keyword arguments.
    If an argument is a list or tuple of the same length as the selection, it will be zipped element-wise. Otherwise, it will be repeated
    Args:
        selection: A list of items to be zipped.
        *args: Additional arguments to be zipped with the selection.
        **kwargs: Additional keyword arguments to be zipped with the selection. for each element in the selection.
    Returns:
        A list of tuples where each tuple contains an element from the selection and its corresponding elements from the additional arguments and keyword arguments.
    """
    expanded_args = []
    for arg in args:
        if isinstance(arg, (list, tuple)) and len(arg) == len(selection):
            expanded_args.append(arg)
        else:
            expanded_args.append(arg * len(selection))

    expanded_kwargs = []
    for key, value in kwargs.items():
        if isinstance(value, (list, tuple)) and len(value) == len(selection):
            expanded_kwargs.append(value)
        else:
            expanded_kwargs.append(value * len(selection))

    zipped_result = zip(selection, *expanded_args, *expanded_kwargs)
    return list(zipped_result)


@beartype
def replace_values(
    data: np.ndarray, values_to_replace: Union[Number, Iterable[Number]], replace_value: Number
) -> np.ndarray:
    """
    Replace one or many values in a Numpy array with a new value.
    Args:
        data: Input data as a numpy array.
        values_to_replace: Values to be replaced with the specified replace value.
        replace_value: Value that will replace the specified old values.
        copy: If the output array is a copy of the input data. Defaults to False.
    Returns:
        Raster data with replaced values.
    """
    return np.where(np.isin(data, values_to_replace), replace_value, data)


@beartype
def replace_values_df(
    df: pd.DataFrame,
    values_to_replace: Union[Number, Iterable[Number]],
    replace_value: Number,
    columns: Optional[Iterable[str]] = None,
    copy: bool = False,
) -> pd.DataFrame:
    """
    Replace one or many values in a DataFrame with a new value.
    Args:
        df: Input data as a DataFrame.
        values_to_replace: Values to be replaced with the specified replace value.
        replace_value: Value that will replace the specified old values.
        columns: Column names to target the replacement. Defaults to None (all columns).
        copy: If the output array is a copy of the input data. Defaults to False.
    Returns:
        DataFrame with replaced values.
    """
    if columns is None:
        columns = df.columns
    else:
        if not check_columns_valid(df, columns):
            raise InvalidColumnException("All selected columns were not found in the input DataFrame.")

    if copy:
        out_df = df.copy()
    else:
        out_df = df

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


@beartype
def cast_array_to_int(
    data: np.ndarray, scalar: Optional[Number] = None, initial_dtype: Optional[np.dtype] = None
) -> np.ndarray:
    """
    Casts an integer array to minimal precision based on both array and scalar value.
    Args:
        data: Input array
        scalar: Input scalar.
    Returns:
        The input array if not of integer type else the casted array with the lowest integer precision possible.
    """
    if initial_dtype is None:
        initial_dtype = data.dtype

    if np.issubdtype(initial_dtype, np.integer):
        data_dtype = get_min_int_type(data)
    else:
        data_dtype = data.dtype

    if scalar is not None:
        scalar_dtype = get_min_int_type(scalar)
        if scalar_dtype == np.float16:
            scalar_dtype = np.float32
        return data.astype(np.result_type(data_dtype, scalar_dtype))
    else:
        return data.astype(data_dtype)


@beartype
def get_max_decimal_points(data: np.ndarray) -> Number:
    """
    Determines the maximum number decimal places within an array.
    Args:
        data: Input array
    Returns:
        The highest number of decimal places contained of a number within the array.
    """
    if np.issubdtype(data.dtype, np.floating):
        decimals = np.zeros_like(data, dtype=int)
        non_integer_mask = np.not_equal(np.mod(data, 1), 0)
        non_integer_values = data[non_integer_mask]
        decimals[non_integer_mask] = -np.floor(np.log10(np.abs(np.fmod(non_integer_values, 1)))).astype(int)
        max_decimals = np.max(decimals)
    else:
        max_decimals = 0

    return max_decimals


@beartype
def cast_array_to_float(
    data: np.ndarray,
    scalar: Optional[Number] = None,
    cast_int: Optional[bool] = None,
    cast_float: Optional[bool] = None,
) -> np.ndarray:
    """
    Casts an array to a desired dtype.
    Args:
        data: Input array.
        scalar: Input scalar.
    Returns:
        The converted input array if a cast option was activated else the unchanged array.
        If cast for integer, dtype float64.
        If cast for floating point, either float32 or float64.
    """
    if cast_int == True and np.issubdtype(data.dtype, np.integer):
        return data.astype(np.float64)
    elif cast_float == True and np.issubdtype(data.dtype, np.floating):
        data_min = np.min(data)
        data_max = np.max(data)

        if np.finfo(np.float32).min <= data_min <= data_max <= np.finfo(np.float32).max:
            data_dtype = np.float32
        else:
            data_dtype = np.float64

        if scalar is None:
            return data.astype(data_dtype)
        else:
            return data.astype(np.result_type(data_dtype, np.min_scalar_type(float(scalar))))
    else:
        return data


@beartype
def truncate_decimal_places(data: np.ndarray | Number, decimal_places: Number) -> np.ndarray | Number:
    """
    Truncates an array or number to a certain number of decimal places.
    Args:
        data: Input array or single numerical value.
        decimal_places: Number of decimal places.
    Returns:
        Truncated array or number.
    """
    return np.trunc(data * 10**decimal_places) / 10**decimal_places


@beartype
def set_max_precision(data: Optional[np.ndarray] = None) -> int:
    """
    Determines the precision for an array.
    Args:
        data: Input array.
    Returns:
        The precision for a certain dtype if array is floating point, else zero. Default is precision for float32.
    """
    if data is not None:
        if np.issubdtype(data.dtype, np.floating):
            return np.finfo(data.dtype).precision
        else:
            return 0
    else:
        return np.finfo(np.float32).precision
