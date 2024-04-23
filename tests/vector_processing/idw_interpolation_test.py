from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio import crs
from shapely.geometry import Point

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidParameterValueException, NonMatchingCrsException
from eis_toolkit.utilities.raster import profile_from_extent_and_pixel_size
from eis_toolkit.vector_processing.idw_interpolation import idw

test_dir = Path(__file__).parent.parent
idw_test_data = test_dir.joinpath("data/remote/interpolating/idw_test_data.tif")


@pytest.fixture
def test_points():
    """Simple test data."""
    data = {
        "value1": [1, 2, 3, 4, 5],
        "value2": [5, 4, 3, 2, 1],
        "geometry": [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3), Point(4, 4)],
    }
    return gpd.GeoDataFrame(data)


@pytest.fixture
def validated_points():
    """Test data."""
    data = {
        "random_number": [124, 248, 496, 992],
        "geometry": [
            Point(24.945831, 60.192059),
            Point(24.6559, 60.2055),
            Point(25.0378, 60.2934),
            Point(24.7284, 60.2124),
        ],
    }
    return gpd.GeoDataFrame(data)


@pytest.fixture
def raster_profile():
    """Raster profile for testing."""
    resolution = (0.0049, 0.0047)
    extent = (24.6558990000000016, 25.0378036000000002, 60.1920590000000004, 60.2934078769999999)
    raster_profile = profile_from_extent_and_pixel_size(extent, resolution, round_strategy="down")
    return raster_profile


@pytest.fixture
def test_empty_gdf():
    """Test empty GeoDataFrame."""
    data = {
        "geometry": [],
        "values": [],
    }
    return gpd.GeoDataFrame(data)


def test_validated_points_with_extent(validated_points, raster_profile):
    """Test IDW."""
    target_column = "random_number"
    interpolated_values = idw(
        geodataframe=validated_points, target_column=target_column, raster_profile=raster_profile, power=2
    )
    assert target_column in validated_points.columns

    with rasterio.open(idw_test_data) as src:
        external_values = src.read(1)

    np.testing.assert_almost_equal(interpolated_values, external_values, decimal=2)


def test_invalid_column(test_points, raster_profile):
    """Test invalid column GeoDataFrame."""
    target_column = "not-in-data-column"
    with pytest.raises(InvalidParameterValueException):
        idw(geodataframe=test_points, target_column=target_column, raster_profile=raster_profile)


def test_empty_geodataframe(test_empty_gdf, raster_profile):
    """Test empty GeoDataFrame."""
    target_column = "values"
    with pytest.raises(EmptyDataFrameException):
        idw(geodataframe=test_empty_gdf, target_column=target_column, raster_profile=raster_profile)


def test_invalid_profile(test_points, raster_profile):
    """Test invalid resolution."""
    target_column = "random_number"
    profile_invalid = raster_profile.copy()
    profile_invalid["transform"] = None
    with pytest.raises(InvalidParameterValueException):
        idw(geodataframe=test_points, target_column=target_column, raster_profile=profile_invalid)


def test_mismatching_crs(test_points, raster_profile):
    """Test invalid resolution."""
    target_column = "random_number"
    profile_invalid = raster_profile.copy()
    profile_invalid["crs"] = crs.CRS.from_epsg(3067)
    with pytest.raises(NonMatchingCrsException):
        idw(geodataframe=test_points, target_column=target_column, raster_profile=profile_invalid)
