import geopandas as gdp
import numpy as np
import pandas as pd
import pytest
import rasterio

from eis_toolkit.exceptions import (
    EmptyDataException,
    EmptyDataFrameException,
    InvalidColumnException,
    InvalidParameterValueException,
)
from eis_toolkit.exploratory_analyses.k_means_cluster import k_means_clustering_array, k_means_clustering_vector
from tests.raster_processing.clip_test import raster_path as SMALL_RASTER_PATH

df = pd.DataFrame(
    {
        "Location": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        "Latitude": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Longitude": [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        "Attribute1": [2, 2, 3, 3, 1, 2, 4, 4, 0, 0],
        "Attribute2": [2, 2, 1, 4, 3, 2, 3, 2, 0, 0],
    }
)
TEST_GDF = gdp.GeoDataFrame(df, geometry=gdp.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326")

with rasterio.open(SMALL_RASTER_PATH) as raster:
    arr1 = raster.read(1)
    TEST_ARRAY = np.stack([arr1, arr1], axis=0)


def test_k_means_clustering_vector_output():
    """Test that k-means vector function assings data points into correct clusters."""
    kmeans_gdf = k_means_clustering_vector(data=TEST_GDF, number_of_clusters=2, random_state=0)
    kmeans_labels = kmeans_gdf["cluster"]
    # For some reason K-means returns the labels reversed in some distributions/platforms
    # Testing simply counts of points beloning to different clusters to for now
    expected_counts = {0: 5, 1: 5}
    counts = kmeans_labels.value_counts()
    np.testing.assert_equal(counts[0], expected_counts[0])
    np.testing.assert_equal(counts[1], expected_counts[1])


def test_k_means_clustering_vector_output_with_columns():
    """Test that k-means vector function assings data points into correct clusters with specified columns."""
    columns = ["Attribute1", "Attribute2"]
    kmeans_gdf = k_means_clustering_vector(
        data=TEST_GDF, include_coordinates=True, columns=columns, number_of_clusters=3, random_state=0
    )
    kmeans_labels = kmeans_gdf["cluster"]

    expected_counts = {0: 4, 1: 4, 2: 2}
    counts = kmeans_labels.value_counts()
    np.testing.assert_equal(counts[0], expected_counts[0])
    np.testing.assert_equal(counts[1], expected_counts[1])
    np.testing.assert_equal(counts[2], expected_counts[2])


def test_k_means_clustering_raster_output():
    """Test that k-means array function assings data points into correct clusters."""
    kmeans_array = k_means_clustering_array(data=TEST_ARRAY, number_of_clusters=2, random_state=0)
    one_count = np.count_nonzero(kmeans_array)
    zero_count = kmeans_array.shape[0] * kmeans_array.shape[1] - one_count

    expected_counts = {0: 1285, 1: 1291}
    np.testing.assert_equal(zero_count, expected_counts[0])
    np.testing.assert_equal(one_count, expected_counts[1])


def test_optimal_number_of_clusters():
    """Test computing finding optimal number of clusters works."""
    kmeans_gdf = k_means_clustering_vector(data=TEST_GDF, random_state=0)
    kmeans_clusters = len(np.unique(kmeans_gdf["cluster"]))
    assert kmeans_clusters == 3


def test_invalid_number_of_clusters():
    """Test that invalid number of clusters raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        k_means_clustering_vector(data=TEST_GDF, number_of_clusters=0)
    with pytest.raises(InvalidParameterValueException):
        k_means_clustering_array(data=TEST_ARRAY, number_of_clusters=0)


def test_empty_input():
    """Test that empty input raises the correct exception."""
    with pytest.raises(EmptyDataFrameException):
        k_means_clustering_vector(gdp.GeoDataFrame(), number_of_clusters=2)
    with pytest.raises(EmptyDataException):
        k_means_clustering_array(np.array([]), number_of_clusters=2)


def test_no_coordinates_no_columns():
    """Test that omitting both coordinates and columns in k-means vector raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        k_means_clustering_vector(TEST_GDF, include_coordinates=False, columns=[])


def test_invalid_columns():
    """Test that specifying missing/invalid columns for k-means vector raises the correct exception."""
    with pytest.raises(InvalidColumnException):
        k_means_clustering_vector(TEST_GDF, columns=["Invalid_column_name"])
