import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import InvalidColumnIndexException
from eis_toolkit.transformations.coda.alr import alr_transform

ONES_DATAFRAME_4x4 = pd.DataFrame(np.ones((4, 4)), columns=["a", "b", "c", "d"])

ZEROS_DATAFRAME_4x3 = pd.DataFrame(np.zeros((4, 3)), columns=["a", "b", "c"])

sample_array = np.array([[65, 12, 18, 5], [63, 16, 15, 6]])
SAMPLE_DATAFRAME = pd.DataFrame(sample_array, columns=["a", "b", "c", "d"])


def test_alr_transform_simple():
    """Test ALR transformation core functionality."""
    result = alr_transform(ONES_DATAFRAME_4x4)
    pd.testing.assert_frame_equal(result, ZEROS_DATAFRAME_4x3)


def test_alr_transform_with_out_of_bounds_denominator_column():
    """Test that providing a column index that is out of bounds raises the correct exception."""
    with pytest.raises(InvalidColumnIndexException):
        alr_transform(SAMPLE_DATAFRAME, 4)
    with pytest.raises(InvalidColumnIndexException):
        alr_transform(SAMPLE_DATAFRAME, -5)


def test_alr_transform_redundant_column():
    """
    Test alr transformation with the keep_redundant_column option set to True.

    Test that the redundant column is found in the result, and that it contains the expected
    values when requesting to keep the redundant column in the alr transformed data.
    """
    idx = -1
    redundant_column = SAMPLE_DATAFRAME.columns[idx]
    result = alr_transform(SAMPLE_DATAFRAME, idx, keep_redundant_column=True)

    assert redundant_column in result.columns
    assert all([val == 0 for val in result.iloc[:, -1].values])


# TODO: test with unnamed columns
