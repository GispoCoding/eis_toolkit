from contextlib import contextmanager
from numbers import Number

import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Any, List, Optional, Sequence, Tuple, Union
from osgeo import gdal
from rasterio import transform
from shapely.geometry import Point

from eis_toolkit.exceptions import InvalidColumnException, InvalidColumnIndexException
from eis_toolkit.utilities.checks.dataframe import check_columns_valid
from eis_toolkit.utilities.checks.parameter import check_dtype_for_int


@beartype
def reduce_ndim(
    data: np.ndarray,
) -> np.ndarray:
    """
    Reduce the number of dimensions of a numpy array.

    Args:
        data: The input raster data as a numpy array.

    Returns:
        The reduced array.
    """
    return np.squeeze(data) if data.ndim >= 3 else data


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


def expand_and_zip(selection: List[Any], *args: Union[List[Any], Tuple[Any]], **kwargs: Any) -> List[Tuple[Any]]:
    """
    Expand and zip a selection with additional arguments and keyword arguments.

    If an argument is a list or tuple of the same length as the selection, it will be zipped element-wise.
    Otherwise, it will be repeated.

    Args:
        selection: A list of items to be zipped.
        *args: Additional arguments to be zipped with the selection.
        **kwargs: Additional keyword arguments to be zipped with the selection. for each element in the selection.

    Returns:
        A list of tuples where each tuple contains an element from the selection and its corresponding
        elements from the additional arguments and keyword arguments.
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
def cast_scalar_to_int(scalar: Number) -> Number:
    """
    Cast a numerical value to integer type if possible.

    Args:
        scalar: Input scalar value.

    Returns:
        The input scalar as an integer if it can be cast, else the original scalar.
    """
    if check_dtype_for_int(scalar) is True:
        return int(scalar)
    else:
        return scalar


def get_min_int_type(data: Union[np.ndarray, Number]) -> np.dtype:
    """
    Check for the lowest integer dtype.

    Args:
        data: Input numpy array or single number.

    Returns:
        The lowest integer dtype possible according to the number(s).
    """
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
    Cast an integer array to minimal precision based on both array and scalar value.

    Args:
        data: Input array.
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
        scalar = cast_scalar_to_int(scalar)

        if isinstance(scalar, int):
            scalar_dtype = get_min_int_type(scalar)
        else:
            if np.min_scalar_type(scalar) == np.float16:
                scalar_dtype = np.float32
        return data.astype(np.result_type(data_dtype, scalar_dtype))
    else:
        return data.astype(data_dtype)


@beartype
def cast_array_to_float(
    data: np.ndarray,
    scalar: Optional[Number] = None,
    cast_int: Optional[bool] = None,
    cast_float: Optional[bool] = None,
) -> np.ndarray:
    """
    Cast an array to a desired dtype.

    Args:
        data: Input array.
        scalar: Input scalar.

    Returns:
        The converted input array if a cast option was activated else the unchanged array.
        If cast for integer, dtype float64.
        If cast for floating point, either float32 or float64.
    """
    if cast_int is True and np.issubdtype(data.dtype, np.integer):
        return data.astype(np.float64)
    elif cast_float is True and np.issubdtype(data.dtype, np.floating):
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
def truncate_decimal_places(data: Union[np.ndarray, Number], decimal_places: int) -> Union[np.ndarray, Number]:
    """
    Truncate an array or number to a certain number of decimal places.

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
    Determine the precision for an array.

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


@beartype
def rename_columns_by_pattern(df: pd.DataFrame, pattern: Optional[str] = None) -> pd.DataFrame:
    """Rename DataFrame columns with a pattern and a running number."""
    columns = [col for col in df.columns]
    pattern = pattern if pattern is not None else "V"
    names = dict()

    for i in range(len(columns)):
        names[columns[i]] = f"{pattern}{i + 1}"

    return df.rename(columns=names)


@beartype
def rename_columns(df: pd.DataFrame, colnames: Sequence[str]) -> pd.DataFrame:
    """
    Replace DataFrame column names with the provided column names.

    Args:
        df: Input DataFrame.
        colnames: A list of column names in order.

    Returns:
        A DataFrame with the column names renamed to the names provided, up to as many are available.

    Raises:
        InvalidColumnIndexException: The amount of provided column names exceeds the amount of columns.
    """
    if len(colnames) > df.shape[1]:
        raise InvalidColumnIndexException()

    columns = [col for col in df.columns]
    names = dict()

    for i in range(len(colnames)):
        names[columns[i]] = colnames[i]

    return df.rename(columns=names)


def row_points(
    row: int,
    cols: np.ndarray,
    raster_transform: transform.Affine,
) -> List[Point]:
    """Transform raster row cells to shapely Points.

    Args:
        row: Row index of cells to transfer
        cols: Array of column indexes to transfer
        raster_transform: Affine transformation matrix of the raster

    Returns:
        List of shapely Points
    """
    # transform.xy accepts either cols or rows as an array. The other then has
    # to be an integer. The resulting x and y point coordinates are therefore
    # in a 1D array

    if len(cols) == 0:
        return []

    point_xs, point_ys = zip(*[raster_transform * (col + 0.5, row + 0.5) for col in cols])
    # point_xs, point_ys = transform.xy(transform=raster_transform, cols=cols, rows=row)
    return [Point(x, y) for x, y in zip(point_xs, point_ys)]


@contextmanager
def toggle_gdal_exceptions():
    """Toggle GDAL exceptions using a context manager.

    If the exceptions are already enabled, this function will do nothing.
    """
    already_has_exceptions_enabled = False
    try:
        if gdal.GetUseExceptions() != 0:
            already_has_exceptions_enabled = True
        gdal.UseExceptions()
        yield
    finally:
        if not already_has_exceptions_enabled:
            gdal.DontUseExceptions()
