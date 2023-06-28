import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidParameterValueException
from eis_toolkit.vector_processing.kriging import kriging

np.random.seed(6)
x = np.random.uniform(0, 5, size=(10, 1))
y = np.random.uniform(0, 5, size=(10, 1))
z = np.random.uniform(0, 2, size=(10, 1))
data = np.hstack((x, y, z))

df = pd.DataFrame(data, columns=["x", "y", "z"])
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"], df["z"]))


def test_ordinary_kriging_output():
    """Test that ordinary kriging output has correct shape and values."""
    z_interpolated = kriging(data=gdf, resolution=(10, 10), limits=[(0, 5), (0, 5)])
    expected_shape = (10, 10)
    expected_value_first_pixel = 1.47416754
    expected_value_last_pixel = 0.73852108
    assert z_interpolated.shape == expected_shape
    assert round(z_interpolated[0][0], 8) == expected_value_first_pixel
    assert round(z_interpolated[9][9], 8) == expected_value_last_pixel


def test_universal_kriging_output():
    """Test that universal kriging output has correct shape and values."""
    z_interpolated = kriging(data=gdf, resolution=(10, 10), limits=[(0, 5), (0, 5)], method="universal_kriging")
    expected_shape = (10, 10)
    expected_value_first_pixel = 2.37243475
    expected_value_last_pixel = 0.12787536
    assert z_interpolated.shape == expected_shape
    assert round(z_interpolated[0][0], 8) == expected_value_first_pixel
    assert round(z_interpolated[9][9], 8) == expected_value_last_pixel


def test_empty_geodataframe():
    """Test that empty geodataframe raises the correct exception."""
    empty_gdf = gpd.GeoDataFrame()
    with pytest.raises(EmptyDataFrameException):
        kriging(data=empty_gdf, resolution=(10, 10), limits=[(0, 5), (0, 5)])


def test_invalid_resolution():
    """Test that invalid resolution raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        kriging(data=gdf, resolution=(0, 0), limits=[(0, 5), (0, 5)])
