import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import InvalidColumnException, InvalidDataShapeException
from eis_toolkit.transformations.normalize import normalize

DF = pd.DataFrame(
    {
        "A": [1, 2, 3, 4, 5],  # Min 1, max 5
        "B": [1, 1, 1, 2, 2],  # Min 1, max 2
        "C": [1, 5, 10, 7, 9],  # Min 1, max 10
    },
    dtype=np.float64,
)
ARRAY_TABULAR = DF.to_numpy()
ARRAY_RASTER = ARRAY_TABULAR
ARRAY_RASTER_3D = np.stack([ARRAY_RASTER, ARRAY_RASTER])


def test_normalize_dataframe():
    """Test that normalization of DataFrame works as expected."""
    normalized_df = normalize(DF)
    assert isinstance(normalized_df, pd.DataFrame)
    np.testing.assert_array_equal(normalized_df["A"].to_numpy(), [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_array_equal(normalized_df["B"].to_numpy(), [0.0, 0.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_almost_equal(normalized_df["C"].to_numpy(), [0.0, 0.4444, 1.0, 0.6667, 0.8889], decimal=3)


def test_noramlize_dataframe_column_selection():
    """Test that normalization of DataFrame with column selection works as expected."""
    normalized_df = normalize(DF, columns=["A", "B"])
    assert isinstance(normalized_df, pd.DataFrame)
    np.testing.assert_array_equal(normalized_df["A"].to_numpy(), [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_array_equal(normalized_df["B"].to_numpy(), [0.0, 0.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(normalized_df["C"].to_numpy(), [1, 5, 10, 7, 9])


def test_normalize_array_tabular():
    """Test that normalization of numpy array with tabular format works as expected."""
    normalized_array = normalize(ARRAY_TABULAR, array_type="tabular")
    assert isinstance(normalized_array, np.ndarray)
    np.testing.assert_equal(normalized_array.ndim, 2)
    np.testing.assert_array_equal(normalized_array[:, 0], [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_array_equal(normalized_array[:, 1], [0.0, 0.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_almost_equal(normalized_array[:, 2], [0.0, 0.4444, 1.0, 0.6667, 0.8889], decimal=3)


def test_normalize_array_raster():
    """Test that normalization of 2D numpy array with raster format works as expected."""
    normalized_array = normalize(ARRAY_RASTER, array_type="raster")
    assert isinstance(normalized_array, np.ndarray)
    np.testing.assert_equal(normalized_array.ndim, 2)
    np.testing.assert_array_almost_equal(
        normalized_array,
        [[0.0, 0.0, 0.0], [0.1111, 0.0, 0.4444], [0.2222, 0.0, 1.0], [0.3333, 0.111, 0.6667], [0.4444, 0.111, 0.8889]],
        3,
    )


def test_normalize_array_raster_3D():
    """Test that normalization of 3D numpy array with raster format works as expected."""
    normalized_array = normalize(ARRAY_RASTER_3D, array_type="raster")
    assert isinstance(normalized_array, np.ndarray)
    np.testing.assert_equal(normalized_array.ndim, 3)
    np.testing.assert_array_almost_equal(
        normalized_array,
        [
            [
                [0.0, 0.0, 0.0],
                [0.1111, 0.0, 0.4444],
                [0.2222, 0.0, 1.0],
                [0.3333, 0.111, 0.6667],
                [0.4444, 0.111, 0.8889],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.1111, 0.0, 0.4444],
                [0.2222, 0.0, 1.0],
                [0.3333, 0.111, 0.6667],
                [0.4444, 0.111, 0.8889],
            ],
        ],
        3,
    )


def test_normalize_dataframe_invalid_column():
    """Test that invalid column selection raises the correct exception."""
    with pytest.raises(InvalidColumnException):
        normalize(DF, columns=["D"])


def test_normalize_array_tabular_invalid_shape():
    """Test that invalid input data shape for tabular format raises the correct exception."""
    with pytest.raises(InvalidDataShapeException):
        normalize(ARRAY_RASTER_3D, array_type="tabular")


def test_normalize_array_raster_invalid_shape():
    """Test that invalid input data shape for raster format raises the correct exception."""
    with pytest.raises(InvalidDataShapeException):
        normalize(ARRAY_RASTER[0], array_type="raster")
