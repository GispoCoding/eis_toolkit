import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import InvalidColumnException, NumericValueSignException
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

    result = alr_transform(df, denominator_column="b", keep_denominator_column=True)
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

    result = alr_transform(df, denominator_column="b")
    expected = pd.DataFrame(
        {"V1": [np.log(0.25), np.log(2)], "V2": [np.log(0.25), np.log(2)], "V3": [np.log(0.25), np.log(2)]},
        dtype=np.float64,
    )
    pd.testing.assert_frame_equal(result, expected)


def test_alr_transform_with_columns():
    """Test ALR transform with column selection."""
    alr = alr_transform(SAMPLE_DATAFRAME, columns=["a", "c", "d"], denominator_column="c", keep_denominator_column=True)

    expected = pd.DataFrame(
        {
            "V1": [np.log(65 / 18), np.log(63 / 15)],
            "V2": [np.log(18 / 18), np.log(15 / 15)],
            "V3": [np.log(5 / 18), np.log(6 / 15)],
        },
        dtype=np.float64,
    )
    pd.testing.assert_frame_equal(alr, expected)


def test_alr_transform_with_invalid_denominator_column():
    """Test that providing a denominator column doesn't exist raises the correct exception."""
    with pytest.raises(InvalidColumnException):
        alr_transform(SAMPLE_DATAFRAME, "e")


def test_alr_transform_with_invalid_columns():
    """Test that providing invalid columns raises the correct exception."""
    with pytest.raises(InvalidColumnException):
        alr_transform(SAMPLE_DATAFRAME, columns=["x", "y", "z"])


def test_alr_transform_denominator_column():
    """
    Test ALR transformation with the keep_denominator_column option set to True.

    Test that the denominator column is found in the result, and that it contains the expected
    values when requesting to keep the denominator column in the ALR transformed data.
    """
    result = alr_transform(SAMPLE_DATAFRAME, keep_denominator_column=True)

    assert result.shape == SAMPLE_DATAFRAME.shape
    assert all([val == 0 for val in result.iloc[:, -1].values])


def test_inverse_alr():
    """Test inverse ALR core functionality."""
    arr = np.array([[np.log(0.25), np.log(0.25), np.log(0.25)], [np.log(2), np.log(2), np.log(2)]])
    df = pd.DataFrame(arr, columns=["V1", "V2", "V3"], dtype=np.float64)
    column_name = "d"
    result = inverse_alr(df, denominator_column=column_name, scale=7)
    expected_arr = np.array([[1, 1, 1, 4], [2, 2, 2, 1]])
    expected = pd.DataFrame(expected_arr, columns=["V1", "V2", "V3", "d"], dtype=np.float64)
    pd.testing.assert_frame_equal(result, expected, atol=1e-2)


def test_inverse_alr_with_existing_denominator_column():
    """Test inverse ALR with data where the denominator column already exists."""
    arr = np.array([[np.log(0.25), np.log(0.25), 0.0, np.log(0.25)], [np.log(2), np.log(2), 0.0, np.log(2)]])
    df = pd.DataFrame(arr, columns=["V1", "V2", "d", "V3"], dtype=np.float64)
    column_name = "d"
    expected_arr = np.array([[1, 1, 4, 1], [2, 2, 1, 2]])
    expected = pd.DataFrame(expected_arr, columns=["V1", "V2", "d", "V3"], dtype=np.float64)

    result = inverse_alr(df, denominator_column=column_name, scale=7)
    pd.testing.assert_frame_equal(result, expected, atol=1e-2)


def test_inverse_alr_with_invalid_scale_value():
    """Test that inverse ALR with an invalid input scale raises the correct exception."""
    arr = np.array([[np.log(0.25), np.log(0.25), np.log(0.25)], [np.log(2), np.log(2), np.log(2)]])
    df = pd.DataFrame(arr, columns=["V1", "V2", "V3"], dtype=np.float64)
    with pytest.raises(NumericValueSignException):
        inverse_alr(df, denominator_column="d", scale=0)
    with pytest.raises(NumericValueSignException):
        inverse_alr(df, denominator_column="d", scale=-7)


def test_inverse_alr_with_invalid_columns():
    """Test that providing invalid columns raises the correct exception."""
    arr = np.array([[np.log(0.25), np.log(0.25), np.log(0.25)], [np.log(2), np.log(2), np.log(2)]])
    df = pd.DataFrame(arr, columns=["V1", "V2", "V3"], dtype=np.float64)
    with pytest.raises(InvalidColumnException):
        inverse_alr(df, columns=["a"], denominator_column="V1")
    with pytest.raises(InvalidColumnException):
        inverse_alr(df, columns=["a", "b", "c"], denominator_column="V1")
