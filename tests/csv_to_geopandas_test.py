from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest


from eis_toolkit.conversions.csv_to_geopandas import csv_to_geopandas
from eis_toolkit.exceptions import InvalidParameterValueException, InvalidWktFormatException, InvalidColumnIndexException

parent_dir = Path(__file__).parent
csv_path = parent_dir.joinpath("data/remote/test.csv")


def test_csv_to_geopandas():
    """Test csv to geopandas conversion using WKT format."""
    indexes = [2]
    target_EPSG = 4326
    gdf = csv_to_geopandas(csv_path, indexes, target_EPSG)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.crs.to_epsg() == target_EPSG

def test_csv_to_geopandas_using_wkt_invalid_parameter_value():
    """Test that index out of range raises correct exception."""
    with pytest.raises(InvalidColumnIndexException):
        indexes = [8]
        target_EPSG = 4326
        gdf = csv_to_geopandas(csv_path, indexes, target_EPSG)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert gdf.crs.to_epsg() == target_EPSG

def test_csv_to_geopandas_invalid_wkt():
    """Test that invalid WKT format raises correct exception."""
    with pytest.raises(InvalidWktFormatException):
        indexes = [3]
        target_EPSG = 4326
        gdf = csv_to_geopandas(csv_path, indexes, target_EPSG)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert gdf.crs.to_epsg() == target_EPSG

def test_csv_to_geopandas_points():
    """Test csv with point features to geopandas conversion using lattitude and longitude."""
    indexes = [5,6]
    target_EPSG = 4326
    gdf = csv_to_geopandas(csv_path, indexes, target_EPSG)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.crs.to_epsg() == target_EPSG

def csv_to_geopandas_invalid_parameter_value():
    """Test that index(es) out of range raises correct exception."""
    with pytest.raises(InvalidColumnIndexException):
        indexes = [9,8]
        target_EPSG = 4326
        gdf = csv_to_geopandas(csv_path, indexes, target_EPSG)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert gdf.crs.to_epsg() == target_EPSG

def csv_to_geopandas_points_invalid_coordinate_values():
    """Test that index(es) with invalid coordinate value(s) raises correct exception."""
    with pytest.raises(InvalidParameterValueException):
        indexes = [3,4]
        target_EPSG = 4326
        gdf = csv_to_geopandas(csv_path, indexes, target_EPSG)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert gdf.crs.to_epsg() == target_EPSG