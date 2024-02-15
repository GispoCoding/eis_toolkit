import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from beartype import beartype
from beartype.typing import Union
from statsmodels.stats import stattools
from statsmodels.stats.weightstats import DescrStatsW

from eis_toolkit.exceptions import InvalidColumnException


def _descriptive_statistics(data: Union[rasterio.io.DatasetReader, pd.DataFrame, gpd.GeoDataFrame]) -> dict:
    statistics = DescrStatsW(data)
    min = np.min(data)
    max = np.max(data)
    mean = statistics.mean
    quantiles = statistics.quantile(probs=[0.25, 0.50, 0.75], return_pandas=False)
    standard_deviation = statistics.std
    if mean == 0:
        relative_standard_deviation = np.nan  # By default this would be set to infinity.
    else:
        relative_standard_deviation = standard_deviation / mean
    skew = stattools.robust_skewness(data)
    results_dict = {
        "min": min,
        "max": max,
        "mean": mean,
        "25%": quantiles[0],
        "50%": quantiles[1],
        "75%": quantiles[2],
        "standard_deviation": standard_deviation,
        "relative_standard_deviation": relative_standard_deviation,
        "skew": skew[0],
    }

    return results_dict


@beartype
def descriptive_statistics_dataframe(input_data: Union[pd.DataFrame, gpd.GeoDataFrame], column: str) -> dict:
    """Generate descriptive statistics from vector data.

    Generates min, max, mean, quantiles(25%, 50% and 75%), standard deviation, relative standard deviation and skewness.

    Args:
        input_data: Data to generate descriptive statistics from.
        column: Specify the column to generate descriptive statistics from.

    Returns:
        The descriptive statistics in previously described order.
    """
    if column not in input_data.columns:
        raise InvalidColumnException
    data = input_data[column]
    statistics = _descriptive_statistics(data)
    return statistics


@beartype
def descriptive_statistics_raster(input_data: rasterio.io.DatasetReader) -> dict:
    """Generate descriptive statistics from raster data.

    Generates min, max, mean, quantiles(25%, 50% and 75%), standard deviation, relative standard deviation and skewness.
    Nodata values are removed from the data before the statistics are computed.

    Args:
        input_data: Data to generate descriptive statistics from.

    Returns:
        The descriptive statistics in previously described order.
    """
    data = input_data.read().flatten()
    nodata_value = input_data.nodata
    data = data[data != nodata_value]
    statistics = _descriptive_statistics(data)
    return statistics
