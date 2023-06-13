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
def cast_dtype_to_int(scalar: Number) -> Number:
    """
    Safely casts a numerical value to integer type if possible.
    Args:
        scalar: Input scalar value.
    Returns:
        The input scalar as an integer if it can be cast, else the original scalar.
    """
    if check_dtype_for_int(scalar) == True:
        return int(scalar)
    else:
        return scalar


@beartype
def replace_values(
    data: np.ndarray, values_to_replace: Union[Number, Iterable[Number]], replace_value: Number, copy: bool = False
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
