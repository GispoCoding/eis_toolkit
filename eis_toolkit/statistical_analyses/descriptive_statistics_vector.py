import geopandas as gpd
from beartype import beartype
from statsmodels.stats import stattools
from statsmodels.stats.weightstats import DescrStatsW


# The core descriptive statistics functionality. Used internally by descriptive_statistics.
def _descriptive_statistics_vector(
    data: gpd.GeoDataFrame,
    column: str,
) -> dict:
    statistics = DescrStatsW(data[column])
    mean = statistics.mean
    quantiles = statistics.quantile(probs=[0.25, 0.50, 0.75], return_pandas=False)
    standard_deviation = statistics.std
    relative_standard_deviation = standard_deviation / mean
    skew = stattools.robust_skewness(data[column])
    results_dict = {
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
def descriptive_statistics_vector(data: gpd.GeoDataFrame, column: str) -> dict:
    """Generate descriptive statistics for selected column from vector data.

    Generates mean, quantiles(25%, 50% and 75%), standard deviation, relative standard deviation and skewness.

    Args:
        dataframe: Dataset to be described.
        column: Name or index of the column to be described.

    Returns:
        The descriptive statistics in previously described order.
    """

    statistics = _descriptive_statistics_vector(data, column)
    return statistics
