import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import (
    EmptyDataFrameException,
    InvalidParameterValueException,
    NotApplicableGeometryTypeException,
)
from eis_toolkit.vector_processing.kriging_interpolation import kriging

np.random.seed(0)
x = np.random.uniform(0, 5, size=(10, 1))
y = np.random.uniform(0, 5, size=(10, 1))
z = np.random.uniform(0, 2, size=(10, 1))
data = np.hstack((x, y, z))
df = pd.DataFrame(data, columns=["x", "y", "z"])
gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df["x"], df["y"], df["z"]))
resolution = (10, 10)
extent = (0, 5, 0, 5)


def test_ordinary_kriging_output():
    """Test that ordinary kriging output has correct shape and values."""
    z_interpolated, _ = kriging(data=gdf, resolution=resolution, extent=extent)
    expected_shape = resolution
    expected_value_first_pixel = 0.4831651
    expected_value_last_pixel = 1.51876801
    assert z_interpolated.shape == expected_shape
    assert round(z_interpolated[0][0], 8) == expected_value_first_pixel
    assert round(z_interpolated[9][9], 8) == expected_value_last_pixel


def test_universal_kriging_output():
    """Test that universal kriging output has correct shape and values."""
    z_interpolated, _ = kriging(data=gdf, resolution=resolution, extent=extent, method="universal")
    expected_shape = resolution
    expected_value_first_pixel = -0.21513566
    expected_value_last_pixel = 1.86445049
    assert z_interpolated.shape == expected_shape
    assert round(z_interpolated[0][0], 8) == expected_value_first_pixel
    assert round(z_interpolated[9][9], 8) == expected_value_last_pixel


def test_empty_geodataframe():
    """Test that empty geodataframe raises the correct exception."""
    empty_gdf = gpd.GeoDataFrame()
    with pytest.raises(EmptyDataFrameException):
        kriging(data=empty_gdf, resolution=resolution, extent=extent)


def test_invalid_resolution():
    """Test that invalid resolution raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        kriging(data=gdf, resolution=(0, 0), extent=extent)


def test_invalid_geometry():
    """Test that invalid geometries raise the correct exception."""
    gdf_invalid_geometry = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df["x"], df["y"]))
    with pytest.raises(NotApplicableGeometryTypeException):
        kriging(data=gdf_invalid_geometry, resolution=resolution, extent=extent)


def test_invalid_variogram_model():
    """Test that invalid variogram model raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        kriging(data=gdf, resolution=resolution, extent=extent, variogram_model="invalid_model")


def test_invalid_method():
    """Test that invalid kriging method raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        kriging(data=gdf, resolution=resolution, extent=extent, method="invalid_method")


def test_invalid_drift_term():
    """Test that invalid drift term raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        kriging(
            data=gdf,
            resolution=resolution,
            extent=extent,
            method="universal",
            drift_terms=["regional_linear", "invalid_drift_term"],
        )
