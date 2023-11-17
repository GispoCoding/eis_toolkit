import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import InvalidColumnException
from eis_toolkit.transformations.coda.clr import CLR_transform

SINGLE_ROW_DATAFRAME = pd.DataFrame(np.array([1, 1, 1, 2])[None], columns=["c1", "c2", "c3", "c4"])

ONES_DATAFRAME_4x4 = pd.DataFrame(np.ones((4, 4)), columns=["c1", "c2", "c3", "c4"])

ZEROS_DATAFRAME_4x4 = pd.DataFrame(np.zeros((4, 4)), columns=["c1", "c2", "c3", "c4"])

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


def test_clr_transform_subcomposition():
    """Test CLR transformation with a subcomposition."""
    subcomposition = ["c1", "c2", "c3"]
    result = CLR_transform(SINGLE_ROW_DATAFRAME, subcomposition)
    pd.testing.assert_frame_equal(pd.DataFrame(np.zeros((1, 3)), columns=subcomposition), result)


def test_clr_transform_subcomposition_single_component():
    """Test CLR transformation with a single component."""
    result = CLR_transform(SINGLE_ROW_DATAFRAME, ["c4"])
    pd.testing.assert_frame_equal(pd.DataFrame(np.zeros((1, 1)), columns=["c4"]), result)


def test_clr_transform_simple():
    """Test CLR transform core functionality."""
    result = CLR_transform(ONES_DATAFRAME_4x4)
    pd.testing.assert_frame_equal(result, ZEROS_DATAFRAME_4x4)


def test_clr_transform_subset_returns_correct_size():
    """Test that the output dataframe contains the same amount of columns as was specified in the parameters."""
    result = CLR_transform(SAMPLE_DATAFRAME, ["c1", "c3", "c4"])
    assert result.shape == (10, 3)


def test_clr_transform_contains_zeros():
    """Test that running the transformation for a dataframe containing zeros raises the correct exception."""
    with pytest.raises(InvalidColumnException):
        df = SAMPLE_DATAFRAME.copy()
        df.iloc[0, 0] = 0
        CLR_transform(df)


def test_clr_transform_with_unexpected_column_name():
    """Test that providing an invalid column name raises the correct exception."""
    with pytest.raises(InvalidColumnException):
        CLR_transform(SAMPLE_DATAFRAME, ["c1", "c5"])


# TODO: test with unnamed columns


# def test_inverse_clr_simple():
#     """TODO: docstring."""
#     result, scale = inverse_CLR(ZEROS_DATAFRAME_4x4)
#     pd.testing.assert_frame_equal(_scale(result, scale), ONES_DATAFRAME_4x4)  # TODO: call each row with its scale


# TODO: test inverse subcomposition
