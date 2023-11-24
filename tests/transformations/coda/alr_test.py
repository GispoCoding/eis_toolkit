import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import (
    InvalidColumnException,
    InvalidColumnIndexException,
    InvalidCompositionException,
    InvalidParameterValueException,
    NumericValueSignException,
)
from eis_toolkit.transformations.coda.alr import alr_transform, alr_transform_old

ONES_DATAFRAME_4x4 = pd.DataFrame(np.ones((4, 4)), columns=["a", "b", "c", "d"])

ZEROS_DATAFRAME_4x3 = pd.DataFrame(np.zeros((4, 3)), columns=["a", "b", "c"])

SAMPLE_DATAFRAME = pd.DataFrame(
    np.array(
        [
            [0.000584, 0.000430, 0.000861, 0.000129],
            [0.000170, 0.000537, 0.000441, 0.000012],
            [0.000286, 0.000365, 0.000131, 0.000009],
            [0.000442, 0.000199, 0.000075, 0.000063],
            [0.000366, 0.000208, 0.000116, 0.000255],
            [0.000310, 0.000041, 0.000219, 0.000086],
            [0.000229, 0.000354, 0.000441, 0.000529],
            [0.000245, 0.000088, 0.000310, 0.000220],
            [0.000317, 0.000446, 0.000946, 0.000090],
            [0.000198, 0.000160, 0.000474, 0.000068],
        ]
    ),
    columns=["a", "b", "c", "d"],
)


def test_alr_transform_simple():
    """Test ALR transformation core functionality."""
    result = alr_transform(ONES_DATAFRAME_4x4)
    pd.testing.assert_frame_equal(result, ZEROS_DATAFRAME_4x3)


def test_alr_transform_contains_zeros():
    """Test that running the transformation for a dataframe containing zeros raises the correct exception."""
    with pytest.raises(NumericValueSignException):
        arr = np.array([[80, 0, 5], [75, 18, 7]])
        df = pd.DataFrame(arr, columns=["a", "b", "c"])
        alr_transform(df)


def test_alr_transform_with_unexpected_column_name():
    """Test that providing an invalid column name raises the correct exception."""
    with pytest.raises(InvalidColumnException):
        alr_transform_old(SAMPLE_DATAFRAME, ["a", "b", "comp3"])


def test_alr_transform_with_out_of_bounds_denominator_column():
    """Test that providing a column index that is out of bounds raises the correct exception."""
    with pytest.raises(InvalidColumnIndexException):
        arr = np.array([[65, 12, 18, 5], [63, 16, 15, 6]])
        df = pd.DataFrame(arr, columns=["a", "b", "c", "d"])
        alr_transform(df, -5)


def test_alr_transform_with_too_few_columns():
    """Test that providing just one column raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        alr_transform_old(SAMPLE_DATAFRAME, ["a"])


def test_alr_transform_redundant_column():
    """
    Test alr transformation with the keep_redundant_column option set to True.

    Test that the redundant column is found in the result, and that it contains the expected
    values when requesting to keep the redundant column in the alr transformed data.
    """
    idx = -1
    arr = np.array([[65, 12, 18, 5], [63, 16, 15, 6]])
    df = pd.DataFrame(arr, columns=["a", "b", "c", "d"])

    redundant_column = df.columns[idx]
    result = alr_transform(df, idx, keep_redundant_column=True)

    assert redundant_column in result.columns
    assert all([val == 0 for val in result.iloc[:, -1].values])


def test_alr_transform_with_nans():
    """Test that running the transformation for a dataframe containing NaN values raises the correct exception."""
    with pytest.raises(InvalidCompositionException):
        df = pd.DataFrame(np.ones((3, 3)), columns=["a", "b", "c"])
        df.iloc[:, 0] = np.NaN
        alr_transform(df)


# TODO: test with unnamed columns
