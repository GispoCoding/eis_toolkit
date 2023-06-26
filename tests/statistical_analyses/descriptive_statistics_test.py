from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import rasterio

from eis_toolkit.exceptions import InvalidColumnException
from eis_toolkit.statistical_analyses.descriptive_statistics import (
    descriptive_statistics_dataframe,
    descriptive_statistics_raster
)

test_dir = Path(__file__).parent.parent
test_csv = pd.read_csv(test_dir.joinpath("data/remote/test.csv"))
test_zero_values = pd.read_csv(test_dir.joinpath("data/remote/test_zero_values.csv"))
test_gpkg = gpd.read_file(test_dir.joinpath("data/remote/test.gpkg"))
src_raster = rasterio.open(test_dir.joinpath("data/remote/small_raster.tif"))


def test_descriptive_statistics_dataframe():
    """Checks that returned statistics are correct when using DataFrame."""
    test = descriptive_statistics_dataframe(test_csv, "random_number")
    assert test["min"] == 124
    assert test["max"] == 31744
    assert test["mean"] == 7040.444444444444
    assert test["25%"] == 496
    assert test["50%"] == 1984
    assert test["75%"] == 7936
    assert test["standard_deviation"] == 9985.87632732808
    assert test["relative_standard_deviation"] == 1.4183587990965332
    assert test["skew"] == 1.6136246052760224


def test_zero_values_column():
    """Test column with all values set to 0."""
    test = descriptive_statistics_dataframe(test_zero_values, "random_number")
    assert test["min"] == 0
    assert test["max"] == 0
    assert test["mean"] == 0
    assert test["25%"] == 0
    assert test["50%"] == 0
    assert test["75%"] == 0
    assert test["standard_deviation"] == 0
    assert pd.isna(test["relative_standard_deviation"]) is True
    assert pd.isna(test["skew"]) is True


def test_invalid_column_name_df():
    """Test that invalid column name raises exception."""
    with pytest.raises(InvalidColumnException):
        descriptive_statistics_dataframe(test_csv, "non_existing_column")


def test_invalid_column_name_gdf():
    """Test that invalid column name raises exception."""
    with pytest.raises(InvalidColumnException):
        descriptive_statistics_dataframe(test_gpkg, "non_existing_column")


def test_descriptive_statistics_geodataframe():
    """Checks that returned statistics are correct when using GeoDataFrame."""
    test = descriptive_statistics_dataframe(test_gpkg, "random_number")
    assert test["min"] == 124
    assert test["max"] == 1984
    assert test["mean"] == 768.8
    assert test["25%"] == 248
    assert test["50%"] == 496
    assert test["75%"] == 992
    assert test["standard_deviation"] == 676.4538121704984
    assert test["relative_standard_deviation"] == 0.8798826901281197
    assert test["skew"] == 0.8890481348169545


def test_descriptive_statistics_raster():
    """Checks that returned statistics are correct when using numpy.ndarray."""
    test = descriptive_statistics_raster(src_raster)
    assert test["min"] == 2.503
    assert test["max"] == 9.67
    assert test["mean"] == 5.186564440993789
    assert test["25%"] == 3.2675
    assert test["50%"] == 5.1825
    assert test["75%"] == 6.0795
    assert test["standard_deviation"] == 1.9646319510650065
    assert test["relative_standard_deviation"] == 0.3787925462830202
    assert test["skew"] == 0.4953143964870621
