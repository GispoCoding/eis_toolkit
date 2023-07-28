import geopandas as gdp
import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidParameterValueException
from eis_toolkit.exploratory_analyses.dbscan import dbscan

np.random.seed(0)
coords = np.random.rand(100, 2) * 2

df = pd.DataFrame(coords, columns=["Latitude", "Longitude"])
gdf = gdp.GeoDataFrame(df, geometry=gdp.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326")


def test_dbscan_output():
    """Test that dbscan output has correct number of clusters and core points."""
    dbscan_gdf = dbscan(data=gdf, max_distance=0.2, min_samples=5)
    dbscan_labels = len(np.unique(dbscan_gdf["cluster"]))
    expected_labels = 8
    expected_core_points = 29
    dbscan_core_points = len(dbscan_gdf[dbscan_gdf["core"] == 1])
    assert dbscan_labels == expected_labels
    assert dbscan_core_points == expected_core_points


def test_invalid_max_distance():
    """Test that invalid maximum distance raises the correct expection."""
    with pytest.raises(InvalidParameterValueException):
        dbscan(data=gdf, max_distance=0.0)


def test_invalid_min_samples():
    """Test that invalid number of minimum samples raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        dbscan(data=gdf, min_samples=1)


def test_empty_geodataframe():
    """Test that empty geodataframe raises the correct exception."""
    empty_gdf = gdp.GeoDataFrame()
    with pytest.raises(EmptyDataFrameException):
        dbscan(data=empty_gdf)
