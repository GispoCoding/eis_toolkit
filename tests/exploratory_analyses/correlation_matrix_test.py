import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from eis_toolkit.exceptions import InvalidParameterValueException, NonNumericDataException
from eis_toolkit.exploratory_analyses.correlation_matrix import correlation_matrix
from tests.exploratory_analyses.covariance_matrix_test import DF, DF_NON_NUMERIC, DF_WITH_NAN


def test_correlation_matrix_nan():
    """Test that returned correlation matrix is correct, when NaN present in the dataframe."""
    expected_correlation_matrix = np.array(
        [
            [1.000000, -0.577350, -1.000000, 1.000000],
            [-0.577350, 1.000000, np.nan, -0.577350],
            [-1.000000, np.nan, 1.000000, -1.000000],
            [1.000000, -0.577350, -1.000000, 1.000000],
        ]
    )
    output_matrix = correlation_matrix(data=DF_WITH_NAN)
    np.testing.assert_array_almost_equal(output_matrix, expected_correlation_matrix)


def test_correlation_matrix():
    """Test that returned correlation matrix is correct."""
    expected_correlation_matrix = np.array(
        [
            [1.000000, -0.577350, -0.904534, 1.000000],
            [-0.577350, 1.000000, 0.174078, -0.577350],
            [-0.904534, 0.174078, 1.000000, -0.904534],
            [1.000000, -0.577350, -0.904534, 1.000000],
        ]
    )
    output_matrix = correlation_matrix(data=DF)
    np.testing.assert_array_almost_equal(output_matrix, expected_correlation_matrix)


def test_correlation_matrix_non_numeric():
    """Test that returned correlation matrix is correct."""
    with pytest.raises(NonNumericDataException):
        correlation_matrix(data=DF_NON_NUMERIC, columns=["a", "b"])


def test_invalid_correlation_method():
    """Test that invalid correlation method raises the correct exception."""
    with pytest.raises(BeartypeCallHintParamViolation):
        correlation_matrix(data=DF, correlation_method="invalid_method")


def test_min_periods_with_kendall():
    """Test that min_periods with correlation_method 'kendall' raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        correlation_matrix(data=DF, correlation_method="kendall", min_periods=1)
