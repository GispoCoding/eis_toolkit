from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio

from eis_toolkit.exceptions import InvalidColumnException
from eis_toolkit.exploratory_analyses.descriptive_statistics import (
    descriptive_statistics_dataframe,
    descriptive_statistics_raster,
)

test_dir = Path(__file__).parent.parent
test_csv = pd.read_csv(test_dir.joinpath("data/remote/test.csv"))
test_zero_values = pd.read_csv(test_dir.joinpath("data/remote/test_zero_values.csv"))
test_gpkg = gpd.read_file(test_dir.joinpath("data/remote/test.gpkg"))
src_raster = rasterio.open(test_dir.joinpath("data/remote/small_raster.tif"))


def test_descriptive_statistics_dataframe():
    """Checks that returned statistics are correct when using DataFrame."""
    test = descriptive_statistics_dataframe(test_csv, "random_number")
    np.testing.assert_almost_equal(test["min"], 124)
    np.testing.assert_almost_equal(test["max"], 31744)
    np.testing.assert_almost_equal(test["mean"], 7040.4444444)
    np.testing.assert_almost_equal(test["25%"], 496)
    np.testing.assert_almost_equal(test["50%"], 1984)
    np.testing.assert_almost_equal(test["75%"], 7936)
    np.testing.assert_almost_equal(test["standard_deviation"], 9985.8763273)
    np.testing.assert_almost_equal(test["relative_standard_deviation"], 1.4183587)
    np.testing.assert_almost_equal(test["skew"], 1.6136246)


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
    np.testing.assert_almost_equal(test["min"], 124)
    np.testing.assert_almost_equal(test["max"], 1984)
    np.testing.assert_almost_equal(test["mean"], 768.8)
    np.testing.assert_almost_equal(test["25%"], 248)
    np.testing.assert_almost_equal(test["50%"], 496)
    np.testing.assert_almost_equal(test["75%"], 992)
    np.testing.assert_almost_equal(test["standard_deviation"], 676.4538121)
    np.testing.assert_almost_equal(test["relative_standard_deviation"], 0.8798826)
    np.testing.assert_almost_equal(test["skew"], 0.8890481)


def test_descriptive_statistics_raster():
    """Checks that returned statistics are correct when using numpy.ndarray."""
    test = descriptive_statistics_raster(src_raster)
    np.testing.assert_almost_equal(test["min"], 2.503)
    np.testing.assert_almost_equal(test["max"], 9.67)
    np.testing.assert_almost_equal(test["mean"], 5.1865644)
    np.testing.assert_almost_equal(test["25%"], 3.2675)
    np.testing.assert_almost_equal(test["50%"], 5.1825)
    np.testing.assert_almost_equal(test["75%"], 6.0795)
    np.testing.assert_almost_equal(test["standard_deviation"], 1.9646319)
    np.testing.assert_almost_equal(test["relative_standard_deviation"], 0.3787925)
    np.testing.assert_almost_equal(test["skew"], 0.4953143)
