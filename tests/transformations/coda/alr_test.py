import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import InvalidColumnIndexException
from eis_toolkit.transformations.coda.alr import alr_transform, inverse_alr

sample_array = np.array([[65, 12, 18, 5], [63, 16, 15, 6]])
SAMPLE_DATAFRAME = pd.DataFrame(sample_array, columns=["a", "b", "c", "d"])


def test_alr_transform_simple():
    """Test ALR transformation core functionality."""
    ones_df_4x4 = pd.DataFrame(np.ones((4, 4)), columns=["a", "b", "c", "d"])
    zeros_df_4x4 = pd.DataFrame(np.zeros((4, 3)), columns=["V1", "V2", "V3"])
    result = alr_transform(ones_df_4x4)
    pd.testing.assert_frame_equal(result, zeros_df_4x4)


def test_alr_transform():
    """Test ALR transformation core functionality."""
    arr = np.array([[1, 4, 1, 1], [2, 1, 2, 2]])
    df = pd.DataFrame(arr, columns=["a", "b", "c", "d"], dtype=np.float64)

    result = alr_transform(df, idx=1, keep_redundant_column=True)
    expected = pd.DataFrame(
        {
            "V1": [np.log(0.25), np.log(2)],
            "V2": [0, 0],
            "V3": [np.log(0.25), np.log(2)],
            "V4": [np.log(0.25), np.log(2)],
        },
        dtype=np.float64,
    )
    pd.testing.assert_frame_equal(result, expected)

    result = alr_transform(df, idx=1)
    expected = pd.DataFrame(
        {"V1": [np.log(0.25), np.log(2)], "V2": [np.log(0.25), np.log(2)], "V3": [np.log(0.25), np.log(2)]},
        dtype=np.float64,
    )
    pd.testing.assert_frame_equal(result, expected)


def test_alr_transform_with_out_of_bounds_denominator_column():
    """Test that providing a column index that is out of bounds raises the correct exception."""
    with pytest.raises(InvalidColumnIndexException):
        alr_transform(SAMPLE_DATAFRAME, 4)
    with pytest.raises(InvalidColumnIndexException):
        alr_transform(SAMPLE_DATAFRAME, -5)


def test_alr_transform_redundant_column():
    """
    Test ALR transformation with the keep_redundant_column option set to True.

    Test that the redundant column is found in the result, and that it contains the expected
    values when requesting to keep the redundant column in the ALR transformed data.
    """
    idx = -1
    result = alr_transform(SAMPLE_DATAFRAME, idx, keep_redundant_column=True)

    assert result.shape == SAMPLE_DATAFRAME.shape
    assert all([val == 0 for val in result.iloc[:, idx].values])


def test_inverse_alr():
    """Test inverse ALR core functionality."""
    arr = np.array([[np.log(0.25), np.log(0.25), np.log(0.25)], [np.log(2), np.log(2), np.log(2)]])
    df = pd.DataFrame(arr, columns=["V1", "V2", "V3"], dtype=np.float64)
    column_name = "d"
    result = inverse_alr(df, column_name, scale=7, idx=-1)
    expected_arr = np.array([[1, 1, 1, 4], [2, 2, 2, 1]])
    expected = pd.DataFrame(expected_arr, columns=["V1", "V2", "V3", "d"], dtype=np.float64)
    pd.testing.assert_frame_equal(result, expected, atol=1e-2)


def test_inverse_alr_column_placement():
    """Test inverse ALR core functionality with nonstandard column placement."""
    arr = np.array([[np.log(0.25), np.log(0.25), np.log(0.25)], [np.log(2), np.log(2), np.log(2)]])
    df = pd.DataFrame(arr, columns=["V1", "V2", "V3"], dtype=np.float64)
    column_name = "d"
    expected_arr = np.array([[1, 1, 4, 1], [2, 2, 1, 2]])
    expected = pd.DataFrame(expected_arr, columns=["V1", "V2", "d", "V3"], dtype=np.float64)

    result = inverse_alr(df, column_name, scale=7, idx=2)
    pd.testing.assert_frame_equal(result, expected, atol=1e-2)

    result = inverse_alr(df, column_name, scale=7, idx=-6)
    pd.testing.assert_frame_equal(result, expected, atol=1e-2)
