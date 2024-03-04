from numbers import Number

import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Dict, Optional, Sequence
from scipy.stats import shapiro

from eis_toolkit.exceptions import (
    EmptyDataException,
    InvalidColumnException,
    InvalidDataShapeException,
    InvalidRasterBandException,
    NonNumericDataException,
    SampleSizeExceededException,
)
from eis_toolkit.utilities.checks.dataframe import check_columns_numeric, check_columns_valid, check_empty_dataframe


@beartype
def normality_test_dataframe(
    data: pd.DataFrame, columns: Optional[Sequence[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute Shapiro-Wilk test for normality on the input DataFrame.

    Nodata values are dropped automatically.

    Args:
        data: Dataframe containing the input data.
        columns: Column selection. If none, normality is tested for all columns.

    Returns:
        Test statistic and p_value for each selected column in a dictionary.

    Raises:
        EmptyDataException: The input data is empty.
        InvalidColumnException: All selected columns were not found in the input data.
        NonNumericDataException: Selected columns contain non-numeric data or no numeric columns were found.
        SampleSizeExceededException: Input data exceeds the maximum of 5000 samples.
    """
    if check_empty_dataframe(data):
        raise EmptyDataException("The input Dataframe is empty.")

    if columns is not None and columns != []:
        if not check_columns_valid(data, columns):
            raise InvalidColumnException("All selected columns were not found in the input DataFrame.")
        if not check_columns_numeric(data, columns):
            raise NonNumericDataException("The selected columns contain non-numeric data.")

        data = data[columns].dropna()

    else:
        columns = data.select_dtypes(include=[np.number]).columns
        if len(columns) == 0:
            raise NonNumericDataException("No numeric columns were found.")

    statistics = {}
    for column in columns:
        if len(data[column]) > 5000:
            raise SampleSizeExceededException(f"Sample size for column '{column}' exceeds the limit of 5000 samples.")
        stat, p_value = shapiro(data[column])
        statistics[column] = {"Statistic": stat, "p-value": p_value}

    return statistics


@beartype
def normality_test_array(
    data: np.ndarray, bands: Optional[Sequence[int]] = None, nodata_value: Optional[Number] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute Shapiro-Wilk test for normality on the input Numpy array.

    It is assumed that 3D input array represents multiband raster and the first dimension is the number of bands
    (same shape as Rasterio reads a raster into an array). Normality is calculated for each band separately.
    NaN values and optionally a specified nodata value are masked out before calculations.

    Args:
        data: Numpy array containing the input data. Array should either be 1D, 2D or 3D.
        bands: Band selection. Applies only if input array is 3D. If None, normality is tested for each band.
        nodata_value: Nodata value to be masked out. Optional parameter.

    Returns:
        Test statistic and p_value for each selected band in a dictionary.

    Raises:
        EmptyDataException: The input data is empty.
        InvalidRasterBandException: All selected bands were not found in the input data.
        InvalidDataShapeException: Input data has incorrect number of dimensions (> 3).
        SampleSizeExceededException: Input data exceeds the maximum of 5000 samples.
    """
    if data.size == 0:
        raise EmptyDataException("The input Numpy array is empty.")

    if data.ndim == 1 or data.ndim == 2:
        prepared_data = np.expand_dims(data, axis=0)
        bands = [1]

    elif data.ndim == 3:
        if bands is not None:
            if not all(band - 1 < len(data) for band in bands):
                raise InvalidRasterBandException("All selected bands were not found in the input array.")
        else:
            bands = range(1, len(data) + 1)
        prepared_data = data

    else:
        raise InvalidDataShapeException(f"The input data has unexpected number of dimensions: {data.ndim}.")

    statistics = {}

    for band in bands:
        band_idx = band - 1
        flattened_data = prepared_data[band_idx].ravel()

        nan_mask = flattened_data == np.nan
        if nodata_value is not None:
            nodata_mask = flattened_data == nodata_value
            nan_mask = nan_mask & nodata_mask
        masked_data = np.ma.masked_array(data=flattened_data, mask=nan_mask)

        if len(masked_data) > 5000:
            raise SampleSizeExceededException(f"Sample size for band '{band}' exceeds the limit of 5000 samples.")

        stat, p_value = shapiro(masked_data)
        statistics[f"Band {band}"] = {"Statistic": stat, "p-value": p_value}

    return statistics
