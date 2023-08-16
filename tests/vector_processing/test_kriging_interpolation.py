import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidParameterValueException
from eis_toolkit.vector_processing.kriging_interpolation import kriging

np.random.seed(0)
x = np.random.uniform(0, 5, size=(10, 1))
y = np.random.uniform(0, 5, size=(10, 1))
z = np.random.uniform(0, 2, size=(10, 1))
data = np.hstack((x, y, z))
df = pd.DataFrame(data, columns=["x", "y", "value"])
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]))
target_column = "value"
resolution = (0.5, 0.5)
extent = (0, 4.5, 0, 4.5)
expected_shape = (10, 10)


def test_ordinary_kriging_output():
    """Test that ordinary kriging output has correct shape and values."""
    z_interpolated, _ = kriging(data=gdf, target_column=target_column, resolution=resolution, extent=extent)
    expected_value_first_pixel = 0.4831651
    expected_value_last_pixel = 1.55695186
    assert z_interpolated.shape == expected_shape
    assert round(z_interpolated[0][0], 8) == expected_value_first_pixel
    assert round(z_interpolated[-1][-1], 8) == expected_value_last_pixel


def test_universal_kriging_output():
    """Test that universal kriging output has correct shape and values."""
    z_interpolated, _ = kriging(
        data=gdf, target_column=target_column, resolution=resolution, extent=extent, method="universal"
    )
    expected_value_first_pixel = -0.21513566
    expected_value_last_pixel = 1.71615605
    assert z_interpolated.shape == expected_shape
    assert round(z_interpolated[0][0], 8) == expected_value_first_pixel
    assert round(z_interpolated[-1][-1], 8) == expected_value_last_pixel


def test_output_without_extent():
    """Test that extent computation works as expected."""
    z_interpolated, _ = kriging(data=gdf, target_column=target_column, resolution=resolution)
    expected_value_first_pixel = 0.45845479
    expected_value_last_pixel = 1.52579066
    assert z_interpolated.shape == (11, 7)
    assert round(z_interpolated[0][0], 8) == expected_value_first_pixel
    assert round(z_interpolated[-1][-1], 8) == expected_value_last_pixel


def test_empty_geodataframe():
    """Test that empty geodataframe raises the correct exception."""
    empty_gdf = gpd.GeoDataFrame()
    with pytest.raises(EmptyDataFrameException):
        kriging(data=empty_gdf, target_column=target_column, resolution=resolution, extent=extent)


def test_invalid_column():
    """Test that invalid column in geodataframe raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        kriging(data=gdf, target_column="invalid_column", resolution=resolution, extent=extent)


def test_invalid_resolution():
    """Test that invalid resolution raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        kriging(data=gdf, target_column=target_column, resolution=(0, 0), extent=extent)


def test_invalid_variogram_model():
    """Test that invalid variogram model raises the correct exception."""
    with pytest.raises(BeartypeCallHintParamViolation):
        kriging(
            data=gdf, target_column=target_column, resolution=resolution, extent=extent, variogram_model="invalid_model"
        )


def test_invalid_method():
    """Test that invalid kriging method raises the correct exception."""
    with pytest.raises(BeartypeCallHintParamViolation):
        kriging(data=gdf, target_column=target_column, resolution=resolution, extent=extent, method="invalid_method")


def test_invalid_drift_term():
    """Test that invalid drift term raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        kriging(
            data=gdf,
            target_column=target_column,
            resolution=resolution,
            extent=extent,
            method="universal",
            drift_terms=["regional_linear", "invalid_drift_term"],
        )
