import sys
from pathlib import Path

import pytest
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
import rasterio
from eis_toolkit import exceptions

from eis_toolkit.vector_processing.idw_interpolation import idw_interpolation

test_dir = Path(__file__).parent.parent
reference_solution_path = test_dir.joinpath("data/remote/idw_test_raster.tif")


@pytest.fixture
def test_points():
    data = {
        'value1': [1, 2, 3, 4, 5],
        'value2': [5, 4, 3, 2, 1],
        'geometry': [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3), Point(4, 4)]
    }
    return gpd.GeoDataFrame(data)


@pytest.fixture
def validated_points():
    data = {
        'random_number': [124, 248, 496, 992],
        'geometry': [Point(24.945831, 60.192059), Point(24.6559, 60.2055),
                     Point(25.0378, 60.2934), Point(24.7284, 60.2124)]
    }
    return gpd.GeoDataFrame(data)


@pytest.fixture
def test_empty_gdf():
    data = {
        "geometry": [],
        "values": [],
    }
    return gpd.GeoDataFrame(data)


def test_validated_points(validated_points):
    target_column = 'random_number'
    resolution = (0.005, 0.005)
    extent = None
    power = 2

    interpolated_values = idw_interpolation(
        geodataframe=validated_points,
        target_column=target_column,
        resolution=resolution,
        extent=extent,
        power=power
    )
    assert target_column in validated_points.columns

    with rasterio.open(reference_solution_path) as src:
        external_values = src.read(1)

    print(f"interpolated_values: {interpolated_values}")
    print(f"external_values: {external_values}")

    np.testing.assert_allclose(interpolated_values, external_values)


def test_validated_points_with_extent(validated_points):
    target_column = 'random_number'
    resolution = (0.005, 0.005)
    extent = (24.655899, 60.192059, 25.037803604, 60.293407876),
    power = 2

    interpolated_values = idw_interpolation(
        geodataframe=validated_points,
        target_column=target_column,
        resolution=resolution,
        extent=extent,
        power=power
    )
    assert target_column in validated_points.columns

    with rasterio.open(reference_solution_path) as src:
        external_values = src.read(1)

    print(f"interpolated_values: {interpolated_values}")
    print(f"external_values: {external_values}")

    np.testing.assert_allclose(interpolated_values, external_values)


def test_invalid_column(test_points):
    target_column = 'not-in-data-column'
    resolution = (1, 1)
    extent = None
    power = 2

    with pytest.raises(exceptions.InvalidParameterValueException):
        idw_interpolation(
            geodataframe=test_points,
            target_column=target_column,
            resolution=resolution,
            extent=extent,
            power=power
        )


def test_empty_geodataframe(test_empty_gdf):
    target_column = 'values'
    resolution = (5, 5)
    extent = None
    power = 5

    with pytest.raises(exceptions.EmptyDataFrameException):
        idw_interpolation(
            geodataframe=test_empty_gdf,
            target_column=target_column,
            resolution=resolution,
            extent=extent,
            power=power
        )


def test_interpolate_vector(test_points):
    target_column = 'value1'
    resolution = (1, 1)
    extent = None
    power = 2

    interpolated_values = idw_interpolation(
        geodataframe=test_points,
        target_column=target_column,
        resolution=resolution,
        extent=extent,
        power=power
    )

    assert target_column in test_points.columns
    interpolated_value = interpolated_values

    # Perform your desired assertions here
    expected_values = np.array([1.0, 1.97913685, 2.59351418, 3.0, 1.97913685, 2.22978488,
                                3.0, 3.40648613, 2.59351418, 3.0, 3.77021482, 4.02086285,
                                3.0, 3.40648613, 4.02086285, 5.0])
    np.testing.assert_allclose(interpolated_value, expected_values, rtol=1e-5, atol=1e-5)
