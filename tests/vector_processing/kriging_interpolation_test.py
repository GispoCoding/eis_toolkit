import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from rasterio import crs, transform

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidParameterValueException, NonMatchingCrsException
from eis_toolkit.utilities.raster import profile_from_extent_and_pixel_size
from eis_toolkit.vector_processing.kriging_interpolation import kriging

np.random.seed(0)
x = np.random.uniform(0, 5, size=(10, 1))
y = np.random.uniform(0, 5, size=(10, 1))
z = np.random.uniform(0, 2, size=(10, 1))
data = np.hstack((x, y, z))
df = pd.DataFrame(data, columns=["x", "y", "value"])
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]))
target_column = "value"

raster_profile = profile_from_extent_and_pixel_size(extent=(0, 5, 0, 5), pixel_size=0.5)
raster_profile["crs"] = gdf.crs
expected_shape = (10, 10)


def test_ordinary_kriging_output():
    """Test that ordinary kriging output has correct shape and values."""
    z_interpolated = kriging(geodataframe=gdf, target_column=target_column, raster_profile=raster_profile)
    expected_value_first_pixel = 0.4216
    expected_value_last_pixel = 1.5815
    assert z_interpolated.shape == expected_shape
    np.testing.assert_almost_equal(z_interpolated[0][0], expected_value_first_pixel, 4)
    np.testing.assert_almost_equal(z_interpolated[-1][-1], expected_value_last_pixel, 4)


def test_universal_kriging_output():
    """Test that universal kriging output has correct shape and values."""
    z_interpolated = kriging(
        geodataframe=gdf, target_column=target_column, raster_profile=raster_profile, method="universal"
    )
    expected_value_first_pixel = -0.2151
    expected_value_last_pixel = 1.7161
    assert z_interpolated.shape == expected_shape
    np.testing.assert_almost_equal(z_interpolated[0][0], expected_value_first_pixel, 4)
    np.testing.assert_almost_equal(z_interpolated[-1][-1], expected_value_last_pixel, 4)


def test_empty_geodataframe():
    """Test that empty geodataframe raises the correct exception."""
    empty_gdf = gpd.GeoDataFrame()
    with pytest.raises(EmptyDataFrameException):
        kriging(geodataframe=empty_gdf, target_column=target_column, raster_profile=raster_profile)


def test_invalid_column():
    """Test that invalid column in geodataframe raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        kriging(geodataframe=gdf, target_column="invalid_column", raster_profile=raster_profile)


def test_mismatching_crs():
    """Test that mismatching CRS of raster profile and geodataframe raises the correct exception."""
    meta = {
        "transform": transform.from_bounds(0, 4.5, 0, 4.5, 10, 10),
        "crs": crs.CRS.from_epsg(3067),
        "width": 10,
    }
    with pytest.raises(NonMatchingCrsException):
        kriging(geodataframe=gdf, target_column=target_column, raster_profile=meta)


def test_invalid_raster_profile():
    """Test that invalid raster metadata/profile raises the correct exception."""
    meta = {
        "transform": transform.from_bounds(0, 4.5, 0, 4.5, 10, 10),
        "crs": gdf.crs,
        "width": 10,
    }
    with pytest.raises(InvalidParameterValueException):
        kriging(geodataframe=gdf, target_column=target_column, raster_profile=meta)


def test_invalid_variogram_model():
    """Test that invalid variogram model raises the correct exception."""
    with pytest.raises(BeartypeCallHintParamViolation):
        kriging(
            geodataframe=gdf,
            target_column=target_column,
            raster_profile=raster_profile,
            variogram_model="invalid_model",
        )


def test_invalid_coordinates_type():
    """Test that invalid coordinates type raises the correct exception."""
    with pytest.raises(BeartypeCallHintParamViolation):
        kriging(
            geodataframe=gdf,
            target_column=target_column,
            raster_profile=raster_profile,
            coordinates_type="invalid_coordinates_type",
        )


def test_invalid_method():
    """Test that invalid kriging method raises the correct exception."""
    with pytest.raises(BeartypeCallHintParamViolation):
        kriging(geodataframe=gdf, target_column=target_column, raster_profile=raster_profile, method="invalid_method")
