import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import (
    EmptyDataException,
    InvalidColumnException,
    InvalidDataShapeException,
    InvalidRasterBandException,
    NonNumericDataException,
    SampleSizeExceededException,
)
from eis_toolkit.exploratory_analyses.normality_test import normality_test_array, normality_test_dataframe

DATA_ARRAY = np.array(
    [
        [[0, 1, 2, 1, 3, 3], [2, 0, 1, 2, 1, 2], [2, 1, 0, 2, 0, 3], [3, 1, 1, 1, 0, 5], [2, 2, 1, 0, 3, 4]],
        [[0, 1, 2, 1, 3, 3], [2, 0, 1, 2, 1, 2], [2, 1, 0, 2, 0, 3], [3, 1, 1, 1, 0, 5], [2, 2, 1, 0, 3, 4]],
    ]
)
DATA_DF = pd.DataFrame(DATA_ARRAY[0], columns=["a", "b", "c", "d", "e", "f"])


def test_normality_test_dataframe():
    """Test that returned normality statistics for DataFrame data are correct."""
    output_statistics = normality_test_dataframe(data=DATA_DF, columns=["a"])
    np.testing.assert_array_almost_equal(output_statistics["a"], (0.82827, 0.13502), decimal=5)


def test_normality_test_array():
    """Test that returned normality statistics for Numpy array data are correct."""
    # 3D array
    output_statistics = normality_test_array(data=DATA_ARRAY, bands=[0])
    np.testing.assert_array_almost_equal(output_statistics[0], (0.91021, 0.01506), decimal=5)

    # 2D array
    output_statistics = normality_test_array(data=DATA_ARRAY[0])
    np.testing.assert_array_almost_equal(output_statistics[0], (0.91021, 0.01506), decimal=5)

    # 1D array
    output_statistics = normality_test_array(data=DATA_ARRAY[0][0])
    np.testing.assert_array_almost_equal(output_statistics[0], (0.9067, 0.41504), decimal=5)


def test_normality_test_dataframe_missing_data():
    """Test that DataFrame input with missing data returns statistics correctly."""
    df_with_nan = DATA_DF.replace(3, np.nan)
    output_statistics = normality_test_dataframe(data=df_with_nan, columns=["a"])
    np.testing.assert_array_almost_equal(output_statistics["a"], (0.62978, 0.00124), decimal=5)


def test_normality_test_array_nodata():
    """Test that Numpy array input with missing data returns statistics correctly."""
    output_statistics = normality_test_array(data=DATA_ARRAY, nodata_value=3)
    np.testing.assert_array_almost_equal(output_statistics[0], (0.91021, 0.01506), decimal=5)


def test_invalid_selection():
    """Test that invalid column names and invalid bands raise the correct exception."""
    with pytest.raises(InvalidColumnException):
        normality_test_dataframe(data=DATA_DF, columns=["g", "h"])

    with pytest.raises(InvalidRasterBandException):
        normality_test_array(data=DATA_ARRAY, bands=[2, 3])


def test_empty_input():
    """Test that empty input raises the correct exception."""
    with pytest.raises(EmptyDataException):
        normality_test_dataframe(data=pd.DataFrame())

    with pytest.raises(EmptyDataException):
        normality_test_array(data=np.array([]))


def test_max_samples():
    """Test that sample count > 5000 raises the correct exception."""
    large_data = np.random.normal(size=5001)
    large_df = pd.DataFrame(large_data, columns=["a"])

    with pytest.raises(SampleSizeExceededException):
        normality_test_dataframe(data=large_df, columns=["a"])

    with pytest.raises(SampleSizeExceededException):
        normality_test_array(data=large_data)


def test_non_numeric_data():
    """Test that non-numeric data for input DataFrame raises the correct exception."""
    with pytest.raises(NonNumericDataException):
        normality_test_dataframe(data=pd.DataFrame(["hey", "there"], columns=["a"]), columns=["a"])


def test_invalid_input_data_shape():
    """Test that invalid shape for input Numpy array raises the correct exception."""
    with pytest.raises(InvalidDataShapeException):
        normality_test_array(data=np.stack([DATA_ARRAY, DATA_ARRAY]))
