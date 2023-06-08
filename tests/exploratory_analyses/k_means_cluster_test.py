import geopandas as gdp
import pandas as pd
import pytest

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidParameterValueException
from eis_toolkit.exploratory_analyses.k_means_cluster import k_means_clustering

df = pd.DataFrame(
    {
        "Location": ["A", "B", "C", "D", "E", "F"],
        "Latitude": [1, 2, 3, 4, 5, 6],
        "Longitude": [-1, -2, -3, -4, -5, -6],
    }
)
gdf = gdp.GeoDataFrame(df, geometry=gdp.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326")


def test_k_means_clustering_output():
    """Test that k-means function assings data points into correct clusters."""
    kmeans_gdf = k_means_clustering(data=gdf, number_of_clusters=2, random_state=0)
    kmeans_labels = kmeans_gdf["cluster"]
    expected_labels = [1, 1, 1, 0, 0, 0]
    assert list(kmeans_labels) == expected_labels


def test_number_of_clusters():
    """Test that number of clusters given as a parameter is non-negative and greater than zero."""
    with pytest.raises(InvalidParameterValueException):
        k_means_clustering(data=gdf, number_of_clusters=0)


def test_empty_geodataframe():
    """Test that empty geodataframe raises the correct exception."""
    empty_gdf = gdp.GeoDataFrame()
    with pytest.raises(EmptyDataFrameException):
        k_means_clustering(data=empty_gdf, number_of_clusters=2)
