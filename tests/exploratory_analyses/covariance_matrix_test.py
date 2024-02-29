import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.exploratory_analyses.covariance_matrix import covariance_matrix

DATA = np.array([[0, 1, 2, 1], [2, 0, 1, 2], [2, 1, 0, 2], [0, 1, 2, 1]])
DF = pd.DataFrame(DATA, columns=["a", "b", "c", "d"])
DF_NON_NUMERIC = pd.DataFrame(
    data=np.array([[0, 1, 2, 1], ["a", "b", "c", "d"], [3, 2, 1, 0], ["c", "d", "b", "a"]]),
    columns=["a", "b", "c", "d"],
)
DF_WITH_NAN = pd.DataFrame(
    data=np.array([[0, 1, 2, 1], [2, 0, np.nan, 2], [2, 1, 0, 2], [0, 1, 2, 1]]), columns=["a", "b", "c", "d"]
)


def test_covariance_matrix_nan():
    """Test that returned covariance matrix is correct, when NaN present in the dataframe."""
    expected_correlation_matrix = np.array(
        [
            [1.333333, -0.333333, -1.333333, 0.666667],
            [-0.333333, 0.25, 0, -0.166667],
            [-1.333333, 0, 1.333333, -0.666667],
            [0.666667, -0.166667, -0.666667, 0.333333],
        ]
    )
    output_matrix = covariance_matrix(data=DF_WITH_NAN)
    np.testing.assert_array_almost_equal(output_matrix, expected_correlation_matrix)


def test_covariance_matrix():
    """Test that returned covariance matrix is correct."""
    expected_covariance_matrix = np.array(
        [
            [1.333333, -0.333333, -1.000000, 0.666667],
            [-0.333333, 0.250000, 0.083333, -0.166667],
            [-1.000000, 0.083333, 0.916667, -0.500000],
            [0.666667, -0.166667, -0.500000, 0.333333],
        ]
    )
    output_matrix = covariance_matrix(data=DF)
    np.testing.assert_array_almost_equal(output_matrix, expected_covariance_matrix)


def test_covariance_matrix_negative_min_periods():
    """Test that negative min_periods value raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        covariance_matrix(data=DF, min_periods=-1)


def test_invalid_ddof():
    """Test that invalid delta degrees of freedom raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        covariance_matrix(data=DF, delta_degrees_of_freedom=-1)
