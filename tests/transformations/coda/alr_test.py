import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import InvalidColumnException, InvalidColumnIndexException, InvalidParameterValueException
from eis_toolkit.transformations.coda.alr import ALR_transform

ONES_DATAFRAME_4x4 = pd.DataFrame(np.ones((4, 4)), columns=["c1", "c2", "c3", "c4"])

ZEROS_DATAFRAME_4x3 = pd.DataFrame(np.zeros((4, 3)), columns=["c1", "c2", "c3"])

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
    columns=["c1", "c2", "c3", "c4"],
)


def test_ALR_transform_simple():
    """Test ALR transformation core functionality."""
    result = ALR_transform(ONES_DATAFRAME_4x4)
    pd.testing.assert_frame_equal(result, ZEROS_DATAFRAME_4x3)


def test_ALR_transform_contains_zeros():
    """Test that running the transformation for a dataframe containing zeros raises the correct exception."""
    with pytest.raises(InvalidColumnException):
        zeros_data = SAMPLE_DATAFRAME.copy()
        zeros_data.iloc[0, 0] = 0
        ALR_transform(zeros_data)


def test_ALR_transform_with_unexpected_column_name():
    """Test that providing an invalid column name raises the correct exception."""
    with pytest.raises(InvalidColumnException):
        ALR_transform(SAMPLE_DATAFRAME, ["c1", "c2", "comp3"])


def test_ALR_transform_with_out_of_bounds_denominator_column():
    """Test that providing a column index that is out of bounds raises the correct exception."""
    with pytest.raises(InvalidColumnIndexException):
        ALR_transform(SAMPLE_DATAFRAME, None, -5)


def test_ALR_transform_with_too_few_columns():
    """Test that providing just one column raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        ALR_transform(SAMPLE_DATAFRAME, ["c1"])


def test_ALR_transform_redundant_column():
    """
    Test ALR transformation with the keep_redundant_column option set to True.

    Test that the redundant column is found in the result, and that it contains the expected
    values when requesting to keep the redundant column in the ALR transformed data.
    """
    cols = ["c2", "c3"]
    idx = -1
    redundant_column = SAMPLE_DATAFRAME.columns[idx]
    result = ALR_transform(SAMPLE_DATAFRAME, cols, -1, keep_redundant_column=True)

    assert redundant_column in result.columns
    assert all([val == 0 for val in result.iloc[:, -1].values])


# TODO: test with unnamed columns
