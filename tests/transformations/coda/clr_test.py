import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import NumericValueSignException
from eis_toolkit.transformations.coda.clr import clr_transform, inverse_clr

sample_array = np.array([[65, 12, 18, 5], [63, 16, 15, 6]], dtype=np.float64)
SAMPLE_DATAFRAME = pd.DataFrame(sample_array, columns=["a", "b", "c", "d"])


def test_clr_transform_simple():
    """Test CLR transform core functionality."""
    ones_df_4x4 = pd.DataFrame(np.ones((4, 4)), columns=["a", "b", "c", "d"])
    zeros_df_4x4 = pd.DataFrame(np.zeros((4, 4)), columns=["V1", "V2", "V3", "V4"])
    result = clr_transform(ones_df_4x4)
    pd.testing.assert_frame_equal(result, zeros_df_4x4)


def test_clr_transform():
    """Test CLR transform core functionality."""
    result = clr_transform(SAMPLE_DATAFRAME)
    expected = pd.DataFrame(
        {"V1": [1.38, 1.29], "V2": [-0.30, -0.08], "V3": [0.10, -0.15], "V4": [-1.18, -1.06]}, dtype=np.float64
    )
    pd.testing.assert_frame_equal(result, expected, atol=1e-2)


def test_clr_transform_with_columns():
    """Test CLR transform with columns."""
    result = clr_transform(SAMPLE_DATAFRAME, columns=["a", "b", "d"])
    expected = pd.DataFrame({"V1": [1.42, 1.24], "V2": [-0.27, -0.13], "V3": [-1.15, -1.11]}, dtype=np.float64)
    pd.testing.assert_frame_equal(result, expected, atol=1e-2)


def test_inverse_clr_simple():
    """Test CLR inverse core functionality."""
    zeros_df_4x4 = pd.DataFrame(np.zeros((4, 4)), columns=["V1", "V2", "V3", "V4"])
    ones_df_4x4 = pd.DataFrame(np.ones((4, 4)), columns=["a", "b", "c", "d"])
    result = inverse_clr(zeros_df_4x4, colnames=["a", "b", "c", "d"], scale=4)
    pd.testing.assert_frame_equal(result, ones_df_4x4)


def test_inverse_clr():
    """Test CLR inverse core functionality."""
    clr = clr_transform(SAMPLE_DATAFRAME)
    result = inverse_clr(clr, colnames=["a", "b", "c", "d"], scale=100)
    pd.testing.assert_frame_equal(result, SAMPLE_DATAFRAME)


def test_inverse_clr_with_columns():
    """Test CLR inverse with columns."""
    clr = clr_transform(SAMPLE_DATAFRAME)
    result = inverse_clr(clr, columns=["V1", "V2"], colnames=["a", "b"], scale=100)
    expected = pd.DataFrame({"a": [84.42, 79.75], "b": [15.58, 20.25]})
    pd.testing.assert_frame_equal(result, expected, atol=1e-2)


def test_inverse_clr_with_invalid_scale_value():
    """Test that inverse CLR with an invalid input scale raises the correct exception."""
    clr = clr_transform(SAMPLE_DATAFRAME)
    with pytest.raises(NumericValueSignException):
        inverse_clr(clr, scale=0)
    with pytest.raises(NumericValueSignException):
        inverse_clr(clr, scale=-1)
