import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from beartype.typing import Optional
from shapely.geometry import Point

from eis_toolkit.exceptions import EmptyDataException, InvalidColumnException, InvalidParameterValueException
from eis_toolkit.exploratory_analyses.pca import compute_pca

DATA = np.array([[1, 1], [2, 2], [3, 3]])
EXPECTED_DATA_PCA_VALUES = expected_pca_values = np.array(
    [[-1.73205081, 1.11022302e-16], [0.0, 0.0], [1.73205081, 1.11022302e-16]]
)
EXPECTED_DATA_COMPONENT_VALUES = np.array([[0.70711, 0.70711], [0.70711, -0.70711]])
EXPECTED_DATA_COMPONENT_VALUES_ALTERNATIVE = np.array([[0.70711, 0.70711], [-0.70711, 0.70711]])
EXPECTED_DATA_EXPLAINED_VARIANCE_RATIOS_VALUES = [1.0, 4.10865055e-33]

DATA_DF = pd.DataFrame(data=DATA, columns=["A", "B"])
EXPECTED_DATA_DF_COLUMNS = ["principal_component_1", "principal_component_2"]

DATA_GDF = gpd.GeoDataFrame(
    data=DATA, columns=["A", "B"], geometry=[Point(1, 2), Point(2, 1), Point(3, 3)], crs="EPSG:4326"
)
EXPECTED_DATA_GDF_COLUMNS = ["principal_component_1", "principal_component_2", "geometry"]

DATA_WITH_NAN = np.array([[1, 1], [2, np.nan], [3, 3]])
DATA_WITH_NODATA = np.array([[1, 1], [2, -9999], [3, 3]])


def _assert_expected_values(
    pca_array: np.ndarray,
    principal_components,
    explained_variances,
    explained_variance_ratios,
    expected_pca_values=EXPECTED_DATA_PCA_VALUES,
    expected_component_values=EXPECTED_DATA_COMPONENT_VALUES,
    expected_component_values_alternative: Optional[np.ndarray] = None,
    expected_explained_variance_ratios_values=EXPECTED_DATA_EXPLAINED_VARIANCE_RATIOS_VALUES,
    decimal_accuracy: int = 5,
    data_shape=DATA.shape,
):
    np.testing.assert_equal(principal_components.size, 4)
    np.testing.assert_equal(explained_variances.size, 2)
    np.testing.assert_equal(explained_variance_ratios.size, 2)
    np.testing.assert_equal(pca_array.shape, data_shape)
    np.testing.assert_array_almost_equal(pca_array, expected_pca_values, decimal=decimal_accuracy)

    try:
        np.testing.assert_array_almost_equal(principal_components, expected_component_values, decimal=decimal_accuracy)
    except AssertionError:
        # Both variations in the sign of the two last members of principal_components occurs
        # depending on environment in nan and nodata tests
        # Both are allowed for those
        if expected_component_values_alternative is None:
            # Deviations in order are not expected unless *_alternative array is passed as input
            raise
        np.testing.assert_array_almost_equal(
            principal_components, expected_component_values_alternative, decimal=decimal_accuracy
        )

    np.testing.assert_array_almost_equal(
        explained_variance_ratios, expected_explained_variance_ratios_values, decimal=decimal_accuracy
    )


@pytest.mark.xfail(sys.platform == "win32", reason="Results deviate on Windows.", raises=AssertionError)
def test_pca_numpy_array():
    """Test that PCA function gives correct output for Numpy array input."""
    pca_array, principal_components, explained_variances, explained_variance_ratios = compute_pca(DATA, 2)

    _assert_expected_values(
        pca_array=pca_array,
        principal_components=principal_components,
        explained_variances=explained_variances,
        explained_variance_ratios=explained_variance_ratios,
    )


@pytest.mark.xfail(sys.platform == "win32", reason="Results deviate on Windows.", raises=AssertionError)
def test_pca_df():
    """Test that PCA function gives correct output for DF input."""

    pca_df, principal_components, explained_variances, explained_variance_ratios = compute_pca(DATA_DF, 2)

    _assert_expected_values(
        pca_array=pca_df.values,
        principal_components=principal_components,
        explained_variances=explained_variances,
        explained_variance_ratios=explained_variance_ratios,
    )
    np.testing.assert_equal(list(pca_df.columns), EXPECTED_DATA_DF_COLUMNS)
    np.testing.assert_equal(pca_df.shape, DATA_DF.shape)


@pytest.mark.xfail(sys.platform == "win32", reason="Results deviate on Windows.", raises=AssertionError)
def test_pca_gdf():
    """Test that PCA function gives correct output for GDF input."""

    pca_gdf, principal_components, explained_variances, explained_variance_ratios = compute_pca(DATA_GDF, 2)

    _assert_expected_values(
        pca_array=pca_gdf.drop(columns=["geometry"]).values,
        principal_components=principal_components,
        explained_variances=explained_variances,
        explained_variance_ratios=explained_variance_ratios,
    )

    np.testing.assert_equal(list(pca_gdf.columns), EXPECTED_DATA_GDF_COLUMNS)
    np.testing.assert_equal(pca_gdf.shape, DATA_GDF.shape)


@pytest.mark.xfail(sys.platform == "win32", reason="Results deviate on Windows.", raises=AssertionError)
def test_pca_with_nan_removal():
    """Test that PCA function gives correct output for Numpy array input that has NaN values and remove strategy."""
    pca_array, principal_components, explained_variances, explained_variance_ratios = compute_pca(
        DATA_WITH_NAN, 2, nodata_handling="remove"
    )

    expected_pca_values = np.array([[-1.414, 0.0], [np.nan, np.nan], [1.414, 0.0]])
    expected_explained_variance_ratios_values = [1.0, 0.0]

    _assert_expected_values(
        pca_array=pca_array,
        principal_components=principal_components,
        explained_variances=explained_variances,
        explained_variance_ratios=explained_variance_ratios,
        expected_pca_values=expected_pca_values,
        # Original implementation expected order as defined by EXPECTED_DATA_COMPONENT_VALUES_ALTERNATIVE
        expected_component_values=EXPECTED_DATA_COMPONENT_VALUES_ALTERNATIVE,
        expected_component_values_alternative=EXPECTED_DATA_COMPONENT_VALUES,
        expected_explained_variance_ratios_values=expected_explained_variance_ratios_values,
        decimal_accuracy=3,
        data_shape=DATA_WITH_NAN.shape,
    )


@pytest.mark.xfail(sys.platform == "win32", reason="Results deviate on Windows.", raises=AssertionError)
def test_pca_with_nan_replace():
    """Test that PCA function gives correct output for Numpy array input that has NaN values and replace strategy."""
    pca_array, principal_components, explained_variances, explained_variance_ratios = compute_pca(
        DATA_WITH_NAN, 2, nodata_handling="replace"
    )

    _assert_expected_values(
        pca_array=pca_array,
        principal_components=principal_components,
        explained_variances=explained_variances,
        explained_variance_ratios=explained_variance_ratios,
        expected_pca_values=EXPECTED_DATA_PCA_VALUES,
        expected_component_values=EXPECTED_DATA_COMPONENT_VALUES,
        expected_component_values_alternative=EXPECTED_DATA_COMPONENT_VALUES_ALTERNATIVE,
        expected_explained_variance_ratios_values=EXPECTED_DATA_EXPLAINED_VARIANCE_RATIOS_VALUES,
        decimal_accuracy=3,
        data_shape=DATA_WITH_NAN.shape,
    )


@pytest.mark.xfail(sys.platform == "win32", reason="Results deviate on Windows.", raises=AssertionError)
def test_pca_with_nodata_removal():
    """Test that PCA function gives correct output for input that has specified nodata values and removal strategy."""
    pca_array, principal_components, explained_variances, explained_variance_ratios = compute_pca(
        DATA_WITH_NODATA, 2, nodata_handling="remove", nodata=-9999
    )

    expected_pca_values = np.array([[-1.414, 0.0], [np.nan, np.nan], [1.414, 0.0]])
    expected_explained_variance_ratios_values = [1.0, 0.0]

    _assert_expected_values(
        pca_array=pca_array,
        principal_components=principal_components,
        explained_variances=explained_variances,
        explained_variance_ratios=explained_variance_ratios,
        expected_pca_values=expected_pca_values,
        # Original implementation expected order as defined by EXPECTED_DATA_COMPONENT_VALUES_ALTERNATIVE
        expected_component_values=EXPECTED_DATA_COMPONENT_VALUES_ALTERNATIVE,
        expected_component_values_alternative=EXPECTED_DATA_COMPONENT_VALUES,
        expected_explained_variance_ratios_values=expected_explained_variance_ratios_values,
        decimal_accuracy=3,
        data_shape=DATA_WITH_NODATA.shape,
    )


def test_pca_empty_data():
    """Test that empty dataframe raises the correct exception."""
    empty_df = pd.DataFrame()
    with pytest.raises(EmptyDataException):
        compute_pca(empty_df, 2)


def test_pca_too_low_number_of_components():
    """Test that invalid (too low) number of PCA components raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        compute_pca(DATA, 0)


def test_pca_too_high_number_of_components():
    """Test that invalid (too high) number of PCA components raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        compute_pca(DATA, 4)


def test_pca_invalid_columns():
    """Test that invalid columns selection raises the correct exception."""
    data_df = pd.DataFrame(data=DATA, columns=["A", "B"])

    with pytest.raises(InvalidColumnException):
        compute_pca(data_df, 2, columns=["A", "C"])
