import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from eis_toolkit import exceptions
from eis_toolkit.exploratory_analyses.pca import compute_pca

parent_dir = Path(__file__).parent
MULTIBAND_RASTER_PATH = parent_dir.joinpath("../data/remote/small_raster_multiband.tif")

DATA = np.array([[1, 1], [2, 2], [3, 3]])


@pytest.mark.xfail(sys.platform == "win32", reason="Results deviate on Windows.", raises=AssertionError)
def test_pca_numpy_array():
    """Test that PCA function gives correct output for Numpy array input."""
    pca_array, explained_variances = compute_pca(DATA, 2)

    expected_pca_values = np.array([[-1.73205081, 1.11022302e-16], [0.0, 0.0], [1.73205081, 1.11022302e-16]])
    expected_explained_variances_values = [1.0, 4.10865055e-33]

    np.testing.assert_equal(explained_variances.size, 2)
    np.testing.assert_equal(pca_array.shape, DATA.shape)

    np.testing.assert_array_almost_equal(pca_array, expected_pca_values, decimal=5)
    np.testing.assert_array_almost_equal(explained_variances, expected_explained_variances_values, decimal=5)


@pytest.mark.xfail(sys.platform == "win32", reason="Results deviate on Windows.", raises=AssertionError)
def test_pca_df():
    """Test that PCA function gives correct output for DF input."""
    data_df = pd.DataFrame(data=DATA, columns=["A", "B"])

    pca_df, explained_variances = compute_pca(data_df, 2)

    expected_columns = ["principal_component_1", "principal_component_2"]
    expected_pca_values = np.array([[-1.73205081, 1.11022302e-16], [0.0, 0.0], [1.73205081, 1.11022302e-16]])
    expected_explained_variances_values = [1.0, 4.10865055e-33]

    np.testing.assert_equal(explained_variances.size, 2)
    np.testing.assert_equal(list(pca_df.columns), expected_columns)
    np.testing.assert_equal(pca_df.shape, data_df.shape)

    np.testing.assert_array_almost_equal(pca_df.values, expected_pca_values, decimal=5)
    np.testing.assert_array_almost_equal(explained_variances, expected_explained_variances_values, decimal=5)


@pytest.mark.xfail(sys.platform == "win32", reason="Results deviate on Windows.", raises=AssertionError)
def test_pca_gdf():
    """Test that PCA function gives correct output for GDF input."""
    data_gdf = gpd.GeoDataFrame(
        data=DATA, columns=["A", "B"], geometry=[Point(1, 2), Point(2, 1), Point(3, 3)], crs="EPSG:4326"
    )

    pca_gdf, explained_variances = compute_pca(data_gdf, 2)

    expected_columns = ["principal_component_1", "principal_component_2", "geometry"]
    expected_pca_values = np.array([[-1.73205081, 1.11022302e-16], [0.0, 0.0], [1.73205081, 1.11022302e-16]])
    expected_explained_variances_values = [1.0, 4.10865055e-33]

    np.testing.assert_equal(explained_variances.size, 2)
    np.testing.assert_equal(list(pca_gdf.columns), expected_columns)
    np.testing.assert_equal(pca_gdf.shape, data_gdf.shape)

    np.testing.assert_array_almost_equal(pca_gdf.drop(columns=["geometry"]).values, expected_pca_values, decimal=5)
    np.testing.assert_array_almost_equal(explained_variances, expected_explained_variances_values, decimal=5)


@pytest.mark.xfail(sys.platform == "win32", reason="Results deviate on Windows.", raises=AssertionError)
def test_pca_with_nan_removal():
    """Test that PCA function gives correct output for Numpy array input that has NaN values and remove strategy."""
    data = np.array([[1, 1], [2, np.nan], [3, 3]])
    pca_array, explained_variances = compute_pca(data, 2, nodata_handling="remove")

    expected_pca_values = np.array([[-1.41421, 0.0], [1.41421, 0.0]])
    expected_explained_variances_values = [1.0, 0.0]

    np.testing.assert_equal(explained_variances.size, 2)
    np.testing.assert_equal(pca_array.shape, (2, 2))

    np.testing.assert_array_almost_equal(pca_array, expected_pca_values, decimal=3)
    np.testing.assert_array_equal(explained_variances, expected_explained_variances_values)


@pytest.mark.xfail(sys.platform == "win32", reason="Results deviate on Windows.", raises=AssertionError)
def test_pca_with_nan_replace():
    """Test that PCA function gives correct output for Numpy array input that has NaN values and replace strategy."""
    data = np.array([[1, 1], [2, np.nan], [3, 3]])
    pca_array, explained_variances = compute_pca(data, 2, nodata_handling="replace")

    expected_pca_values = np.array([[-1.73205, 1.11022e-16], [0, 0], [1.73205, 1.11022e-16]])
    expected_explained_variances_values = [1.0, 4.10865e-33]

    np.testing.assert_equal(explained_variances.size, 2)
    np.testing.assert_equal(pca_array.shape, DATA.shape)

    np.testing.assert_array_almost_equal(pca_array, expected_pca_values, decimal=3)
    np.testing.assert_array_almost_equal(explained_variances, expected_explained_variances_values, decimal=3)


@pytest.mark.xfail(sys.platform == "win32", reason="Results deviate on Windows.", raises=AssertionError)
def test_pca_with_nodata_removal():
    """Test that PCA function gives correct output for input that has specified nodata values and removal strategy."""
    data = np.array([[1, 1], [2, -9999], [3, 3]])
    pca_array, explained_variances = compute_pca(data, 2, nodata_handling="remove", nodata=-9999)

    expected_pca_values = np.array([[-1.41421, 0.0], [1.41421, 0.0]])
    expected_explained_variances_values = [1.0, 0.0]

    np.testing.assert_equal(explained_variances.size, 2)
    np.testing.assert_equal(pca_array.shape, (2, 2))

    np.testing.assert_array_almost_equal(pca_array, expected_pca_values, decimal=3)
    np.testing.assert_array_equal(explained_variances, expected_explained_variances_values)


def test_pca_empty_data():
    """Test that empty dataframe raises the correct exception."""
    empty_df = pd.DataFrame()
    with pytest.raises(exceptions.EmptyDataException):
        compute_pca(empty_df, 2)


def test_pca_too_low_number_of_components():
    """Test that invalid (too low) number of PCA components raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        compute_pca(DATA, 0)


def test_pca_too_high_number_of_components():
    """Test that invalid (too high) number of PCA components raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        compute_pca(DATA, 4)


def test_pca_invalid_columns():
    """Test that invalid columns selection raises the correct exception."""
    data_df = pd.DataFrame(data=DATA, columns=["A", "B"])

    with pytest.raises(exceptions.InvalidColumnException):
        compute_pca(data_df, 2, columns=["A", "C"])
