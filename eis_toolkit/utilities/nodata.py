from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def set_nodata_raster_meta(raster_meta: Dict, nodata_value: float) -> Dict:
    """
    Set new NoData value for raster metadata.

    Args:
        raster_meta (dict): Raster metadata.
        nodata_value (float): NoData value to be set.

    Returns:
        raster_meta (dict): Raster metadata with updated NoData.
    """
    raster_meta.update({"nodata": nodata_value})
    return raster_meta


def replace_values_with_nodata_multiple(data: np.ndarray, values_to_replace: List[float]):
    pass


def replace_values_with_nodata(data: np.ndarray, values_to_replace: List[float], new_nodata: float = np.nan):
    """
    Replace multiple NoData values in a raster numpy array with a new NoData value.

    Args:
        in_data (np.ndarray): Input raster data as a numpy array.
        values_to_replace (List[float]): List of values to be replaced with new_nodata.
        new_nodata (float): New NoData value to be set. Defaults to np.nan.

    Returns:
        np.ndarray: Raster data with updated NoData values.
    """
    out_data = data.copy()
    mask = np.isin(data, values_to_replace)
    out_data[mask] = new_nodata
    return out_data


def replace_nodata_dataframe(
    data: pd.DataFrame, new_nodata: float, old_nodata: float = np.nan, columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Replace the NoData value in specified columns of a DataFrame.

    Args:
        in_data (pd.DataFrame): Input DataFrame data.
        new_nodata (float): New NoData value to be set.
        old_nodata (float): Current NoData value to be replaced. Defaults to np.nan.
        columns (Optional[List[str]]): List of column names to replace NoData values. Defaults to None (all columns).

    Returns:
        pd.DataFrame: DataFrame with updated NoData values.
    """
    out_data = data.copy()
    if columns is None:
        columns = out_data.columns

    for col in columns:
        out_data[col] = out_data[col].replace(old_nodata, new_nodata)

    return out_data
