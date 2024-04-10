import geopandas as gdp
import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import (
    EmptyDataException,
    EmptyDataFrameException,
    InvalidColumnException,
    InvalidDataShapeException,
    InvalidParameterValueException,
)
from eis_toolkit.exploratory_analyses.dbscan import dbscan_array, dbscan_vector
from tests.exploratory_analyses.k_means_cluster_test import TEST_ARRAY

np.random.seed(0)
df = pd.DataFrame(data=np.random.rand(100, 2) * 2, columns=["Latitude", "Longitude"])
df["Attribute1"] = np.random.rand(100)
df["Attribute2"] = np.random.rand(100)
TEST_GDF = gdp.GeoDataFrame(df, geometry=gdp.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326")


def test_dbscan_vector_output():
    """Test that DBSCAN vector output has correct number of clusters."""
    out_gdf = dbscan_vector(data=TEST_GDF, max_distance=0.2, min_samples=5)
    dbscan_labels = len(np.unique(out_gdf["cluster"]))
    expected_labels = 8
    assert dbscan_labels == expected_labels


def test_dbscan_vector_output_with_columns():
    """Test that DBSCAN vector output has correct number of clusters with specified columns."""
    columns = ["Attribute1", "Attribute2"]
    out_gdf = dbscan_vector(data=TEST_GDF, columns=columns, max_distance=0.5, min_samples=5)
    dbscan_labels = len(np.unique(out_gdf["cluster"]))
    expected_labels = 3
    assert dbscan_labels == expected_labels


def test_dbscan_array_output():
    """Test that DBSCAN array output has correct number of clusters."""
    out_array = dbscan_array(data=TEST_ARRAY, max_distance=0.07, min_samples=5)
    dbscan_labels = len(np.unique(out_array))
    expected_labels = 7
    assert dbscan_labels == expected_labels


def test_invalid_max_distance():
    """Test that invalid maximum distance raises the correct expection."""
    with pytest.raises(InvalidParameterValueException):
        dbscan_array(data=TEST_ARRAY, max_distance=0.0)
    with pytest.raises(InvalidParameterValueException):
        dbscan_vector(data=TEST_GDF, max_distance=0.0)


def test_invalid_min_samples():
    """Test that invalid number of minimum samples raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        dbscan_array(data=TEST_ARRAY, min_samples=1)
    with pytest.raises(InvalidParameterValueException):
        dbscan_vector(data=TEST_GDF, min_samples=1)


def test_empty_input():
    """Test that empty input raises the correct exception."""
    with pytest.raises(EmptyDataException):
        dbscan_array(np.array([]))
    with pytest.raises(EmptyDataFrameException):
        dbscan_vector(gdp.GeoDataFrame())


def test_invalid_columns():
    """Test that specifying missing/invalid columns for DBSCAN vector raises the correct exception."""
    with pytest.raises(InvalidColumnException):
        dbscan_vector(TEST_GDF, columns=["Invalid_column_name"])


def test_invalid_array_shape():
    """Test that invalid input array shape raises the correct exception."""
    with pytest.raises(InvalidDataShapeException):
        dbscan_array(TEST_ARRAY[0])
