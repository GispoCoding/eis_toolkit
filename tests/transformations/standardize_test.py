import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import InvalidColumnException, InvalidDataShapeException
from eis_toolkit.transformations.standardize import standardize

DF = pd.DataFrame(
    {
        "A": [1, 2, 3, 4, 5],  # Mean 3, std sqrt(2)
        "B": [1, 1, 1, 2, 2],  # Mean 1.4, std sqrt(1.4)
        "C": [1, 5, 10, 7, 9],  # Mean 6.4, std sqrt(51.2)
    },
    dtype=np.float64,
)
ARRAY_TABULAR = DF.to_numpy()
ARRAY_RASTER = ARRAY_TABULAR
ARRAY_RASTER_3D = np.stack([ARRAY_RASTER, ARRAY_RASTER])


def test_standardize_dataframe():
    """Test that standardization of DataFrame works as expected."""
    standardized_df = standardize(DF)
    assert isinstance(standardized_df, pd.DataFrame)
    np.testing.assert_array_almost_equal(
        standardized_df["A"].to_numpy(), [-1.4142, -0.7071, 0.0, 0.7071, 1.4142], decimal=3
    )
    np.testing.assert_array_almost_equal(
        standardized_df["B"].to_numpy(), [-0.816, -0.816, -0.816, 1.225, 1.225], decimal=3
    )
    np.testing.assert_array_almost_equal(
        standardized_df["C"].to_numpy(), [-1.688, -0.438, 1.125, 0.187, 0.812], decimal=3
    )


def test_noramlize_dataframe_column_selection():
    """Test that standardization of DataFrame with column selection works as expected."""
    standardized_df = standardize(DF, columns=["A", "B"])
    assert isinstance(standardized_df, pd.DataFrame)
    np.testing.assert_array_almost_equal(
        standardized_df["A"].to_numpy(), [-1.4142, -0.7071, 0.0, 0.7071, 1.4142], decimal=3
    )
    np.testing.assert_array_almost_equal(
        standardized_df["B"].to_numpy(), [-0.816, -0.816, -0.816, 1.225, 1.225], decimal=3
    )
    np.testing.assert_array_equal(standardized_df["C"].to_numpy(), [1, 5, 10, 7, 9])


def test_standardize_array_tabular():
    """Test that standardization of numpy array with tabular format works as expected."""
    standardized_array = standardize(ARRAY_TABULAR, array_type="tabular")
    assert isinstance(standardized_array, np.ndarray)
    np.testing.assert_equal(standardized_array.ndim, 2)
    np.testing.assert_array_almost_equal(standardized_array[:, 0], [-1.4142, -0.7071, 0.0, 0.7071, 1.4142], decimal=3)
    np.testing.assert_array_almost_equal(standardized_array[:, 1], [-0.816, -0.816, -0.816, 1.225, 1.225], decimal=3)
    np.testing.assert_array_almost_equal(standardized_array[:, 2], [-1.688, -0.438, 1.125, 0.187, 0.812], decimal=3)


def test_standardize_array_raster():
    """Test that standardization of 2D numpy array with raster format works as expected."""
    standardized_array = standardize(ARRAY_RASTER, array_type="raster")
    assert isinstance(standardized_array, np.ndarray)
    np.testing.assert_equal(standardized_array.ndim, 2)
    np.testing.assert_array_almost_equal(
        standardized_array,
        [
            [-0.8914, -0.8914, -0.8914],
            [-0.5485, -0.8914, 0.4800],
            [-0.2057, -0.8914, 2.1943],
            [0.1371, -0.5486, 1.1657],
            [0.4800, -0.5486, 1.8515],
        ],
        3,
    )


def test_standardize_array_raster_3D():
    """Test that standardization of 3D numpy array with raster format works as expected."""
    standardized_array = standardize(ARRAY_RASTER_3D, array_type="raster")
    print(ARRAY_RASTER_3D)
    print(standardized_array)
    assert isinstance(standardized_array, np.ndarray)
    np.testing.assert_equal(standardized_array.ndim, 3)
    np.testing.assert_array_almost_equal(
        standardized_array,
        [
            [
                [-0.8914, -0.8914, -0.8914],
                [-0.5485, -0.8914, 0.4800],
                [-0.2057, -0.8914, 2.1943],
                [0.1371, -0.5486, 1.1657],
                [0.4800, -0.5486, 1.8515],
            ],
            [
                [-0.8914, -0.8914, -0.8914],
                [-0.5485, -0.8914, 0.4800],
                [-0.2057, -0.8914, 2.1943],
                [0.1371, -0.5486, 1.1657],
                [0.4800, -0.5486, 1.8515],
            ],
        ],
        3,
    )


def test_standardize_dataframe_invalid_column():
    """Test that invalid column selection raises the correct exception."""
    with pytest.raises(InvalidColumnException):
        standardize(DF, columns=["D"])


def test_standardize_array_tabular_invalid_shape():
    """Test that invalid input data shape for tabular format raises the correct exception."""
    with pytest.raises(InvalidDataShapeException):
        standardize(ARRAY_RASTER_3D, array_type="tabular")


def test_standardize_array_raster_invalid_shape():
    """Test that invalid input data shape for raster format raises the correct exception."""
    with pytest.raises(InvalidDataShapeException):
        standardize(ARRAY_RASTER[0], array_type="raster")
