from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import Point

from eis_toolkit import exceptions
from eis_toolkit.vector_processing.simple_idw import simple_idw

test_dir = Path(__file__).parent.parent
extent_set = test_dir.joinpath("data/remote/idw_extent.tif")
no_extent = test_dir.joinpath("data/remote/idw_no_extent.tif")


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
def test_empty_gdf():
    """Test empty GeoDataFrame."""
    data = {
        "geometry": [],
        "values": [],
    }
    return gpd.GeoDataFrame(data)


def test_validated_points(validated_points):
    """Test IDW without extent set."""
    target_column = "random_number"
    resolution = (0.0049, 0.0047)
    extent = None
    power = 2

    interpolated_values = simple_idw(
        geodataframe=validated_points, target_column=target_column, resolution=resolution, extent=extent, power=power
    )
    assert target_column in validated_points.columns

    with rasterio.open(no_extent) as src:
        external_values = src.read(1)

    np.testing.assert_allclose(interpolated_values[2], external_values)


def test_validated_points_with_extent(validated_points):
    """Test IDW with extent set."""
    target_column = "random_number"
    resolution = (0.0049, 0.0047)
    extent = (24.6558990000000016, 60.1920590000000004, 25.0378036000000002, 60.2934078769999999)
    power = 2

    interpolated_values = simple_idw(
        geodataframe=validated_points, target_column=target_column, resolution=resolution, extent=extent, power=power
    )
    assert target_column in validated_points.columns

    with rasterio.open(extent_set) as src:
        external_values = src.read(1)

    np.testing.assert_allclose(interpolated_values[2], external_values)


def test_invalid_column(test_points):
    """Test invalid column GeoDataFrame."""
    target_column = "not-in-data-column"
    resolution = (1, 1)
    extent = None
    power = 2

    with pytest.raises(exceptions.InvalidParameterValueException):
        simple_idw(
            geodataframe=test_points, target_column=target_column, resolution=resolution, extent=extent, power=power
        )


def test_empty_geodataframe(test_empty_gdf):
    """Test empty GeoDataFrame."""
    target_column = "values"
    resolution = (5, 5)
    extent = None
    power = 5

    with pytest.raises(exceptions.EmptyDataFrameException):
        simple_idw(
            geodataframe=test_empty_gdf, target_column=target_column, resolution=resolution, extent=extent, power=power
        )


def test_interpolate_vector(test_points):
    """Test IDW with simple data."""
    target_column = "value1"
    resolution = (1, 1)
    extent = None
    power = 2

    interpolated_values = simple_idw(
        geodataframe=test_points, target_column=target_column, resolution=resolution, extent=extent, power=power
    )

    assert target_column in test_points.columns
    interpolated_value = interpolated_values[2]

    expected_values = np.array(
        [
            [3, 3.40648594, 4.02086331, 5],
            [2.59351406, 3, 3.77021471, 4.02086331],
            [1.97913669, 2.22978529, 3, 3.40648594],
            [1, 1.97913669, 2.59351406, 3],
        ]
    )
    np.testing.assert_allclose(interpolated_value, expected_values, rtol=1e-5, atol=1e-5)
