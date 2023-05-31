import rasterio
from beartype import beartype
from statsmodels.stats import stattools
from statsmodels.stats.weightstats import DescrStatsW


# The core descriptive statistics functionality. Used internally by descriptive_statistics.
def _descriptive_statistics_raster(raster: rasterio.io.DatasetReader) -> dict:

    data = raster.read().flatten()
    statistics = DescrStatsW(data)
    mean = statistics.mean
    quantiles = statistics.quantile(probs=[0.25, 0.50, 0.75], return_pandas=False)
    standard_deviation = statistics.std
    relative_standard_deviation = standard_deviation / mean
    skew = stattools.robust_skewness(data)
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
def descriptive_statistics_raster(raster: rasterio.io.DatasetReader) -> dict:
    """Generate descriptive statistics from raster data.

    Generates mean, quantiles(25%, 50% and 75%), standard deviation, relative standard deviation and skewness.

    Args:
        raster: Raster to generate descriptive statistics from.

    Returns:
        The descriptive statistics in previously described order.
    """

    statistics = _descriptive_statistics_raster(raster)
    return statistics
