from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio

from eis_toolkit.statistical_analyses.descriptive_statistics_raster import descriptive_statistics_raster
from eis_toolkit.statistical_analyses.descriptive_statistics_vector import descriptive_statistics_vector
from tests.clip_test import raster_path as SMALL_RASTER_PATH

test_dir = Path(__file__).parent.parent
test_csv = pd.read_csv(test_dir.joinpath("data/remote/test.csv"))
test_gpkg = gpd.read_file(test_dir.joinpath("data/remote/test.gpkg"))
src_raster = rasterio.open(SMALL_RASTER_PATH)


def test_descriptive_statistics_dataframe():
    """Checks that returned statistics are correct when using DataFrame."""
    test = descriptive_statistics_vector(test_csv, "random_number")
    assert test["mean"] == 7040.444444444444
    assert test["25%"] == 496
    assert test["50%"] == 1984
    assert test["75%"] == 7936
    assert test["standard_deviation"] == 9985.87632732808
    assert test["relative_standard_deviation"] == 1.4183587990965332
    assert test["skew"] == 1.6136246052760224


def test_descriptive_statistics_geodataframe():
    """Checks that returned statistics are correct when using GeoDataFrame."""
    test = descriptive_statistics_vector(test_gpkg, "random_number")
    assert test["mean"] == 768.8
    assert test["25%"] == 248
    assert test["50%"] == 496
    assert test["75%"] == 992
    assert test["standard_deviation"] == 676.4538121704984
    assert test["relative_standard_deviation"] == 0.8798826901281197
    assert test["skew"] == 0.8890481348169545


def test_descriptive_statistics_numpy_ndarray():
    """Checks that returned statistics are correct when using numpy.ndarray."""
    test = descriptive_statistics_raster(src_raster)
    assert test["mean"] == 5.186564440993789
    assert test["25%"] == 3.2675
    assert test["50%"] == 5.1825
    assert test["75%"] == 6.0795
    assert test["standard_deviation"] == 1.9646319510650065
    assert test["relative_standard_deviation"] == 0.3787925462830202
    assert test["skew"] == 0.4953143964870621
