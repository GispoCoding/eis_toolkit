import sys
from pathlib import Path

import pytest
import numpy as np
from shapely.geometry import Point, MultiPoint
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import profiles, transform
from eis_toolkit import exceptions

from eis_toolkit.vector_processing.interpolate_vector_idw import interpolate_vector

test_dir = Path(__file__).parent.parent
reference_solution_path = test_dir.joinpath("data/remote/interpolated_points.tif")


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
        'random_number': [1, 2, 3, 4],
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
    with rasterio.open(reference_solution_path) as src:
        pixel_width, pixel_height = src.res
    target_column = 'random_number'
    power = 2
    base_raster_profile = profiles.Profile(
        dict(width=9, height=6, transform=transform.from_bounds(
            24.3914739999999988, 59.9771080000000012, 25.1934639999999987, 60.5424018579999981,
            width=9, height=6))
    )

    interpolated_values = interpolate_vector(
        geodataframe=validated_points,
        resolution=(pixel_width, pixel_height),
        target_column=target_column,
        power=power,
        base_raster_profile=base_raster_profile
    )
    assert target_column in validated_points.columns

    with rasterio.open(reference_solution_path) as src:
        external_values = src.read(1)

    # Reshape interpolated_values dynamically based on external_values shape
    #  interpolated_values = interpolated_values.reshape(external_values.shape)
    print(f"interpolated_values: {interpolated_values}")
    print(f"external_values: {external_values}")

    np.testing.assert_allclose(interpolated_values, external_values, rtol=1e-5, atol=1e-5)


def test_invalid_column(test_points):
    resolution = 1.0
    target_column = 'not-in-data-column'
    power = 2
    base_raster_profile = None

    with pytest.raises(exceptions.InvalidParameterValueException):
        interpolate_vector(
            geodataframe=test_points,
            resolution=resolution,
            target_column=target_column,
            power=power,
            base_raster_profile=base_raster_profile
        )


def test_empty_geodataframe(test_empty_gdf):
    resolution = 5.0
    target_column = 'values'
    power = 5
    base_raster_profile = None

    with pytest.raises(exceptions.EmptyDataFrameException):
        interpolate_vector(
            geodataframe=test_empty_gdf,
            resolution=resolution,
            target_column=target_column,
            power=power,
            base_raster_profile=base_raster_profile
        )


def test_interpolate_vector(test_points):
    resolution = 1.0
    target_column = 'value1'
    power = 2
    base_raster_profile = None

    interpolated_values = interpolate_vector(
        geodataframe=test_points,
        resolution=resolution,
        target_column=target_column,
        power=power,
        base_raster_profile=base_raster_profile
    )

    assert target_column in test_points.columns
    interpolated_value = interpolated_values

    # Perform your desired assertions here
    expected_values = np.array([1.0, 1.97913685, 2.59351418, 3.0, 1.97913685, 2.22978488,
                                3.0, 3.40648613, 2.59351418, 3.0, 3.77021482, 4.02086285,
                                3.0, 3.40648613, 4.02086285, 5.0])
    np.testing.assert_allclose(interpolated_value, expected_values, rtol=1e-5, atol=1e-5)


def test_interpolate_vector_with_raster_profile(test_points):
    resolution = 1.0
    target_column = 'value1'
    power = 2
    base_raster_profile = profiles.Profile(
        dict(height=4, width=4, transform=transform.from_bounds(1, 1, 5, 5, width=5, height=5))
    )

    interpolated_values = interpolate_vector(
        geodataframe=test_points,
        resolution=resolution,
        target_column=target_column,
        power=power,
        base_raster_profile=base_raster_profile
    )

    assert target_column in test_points.columns
    interpolated_value = interpolated_values

    expected_values = np.array([2.43438794, 2.95383521, 3.26941288, 3.43566589,
                               2.99313871, 3.38950998, 3.71685399, 3.80775634,
                               3.36943011, 3.98582218, 4.30897436, 4.29193548,
                               3.63483607, 4.18160691, 4.95911106, 4.58061304])

    np.testing.assert_allclose(interpolated_value, expected_values, rtol=1e-5, atol=1e-5)
