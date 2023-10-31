import numpy as np
import pandas as pd
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from eis_toolkit import exceptions
from eis_toolkit.statistical_analyses.statistical_tests import (
    chi_square_test,
    correlation_matrix,
    covariance_matrix,
    normality_test,
)

data = np.array([[0, 1, 2, 1], [2, 0, 1, 2], [2, 1, 0, 2], [0, 1, 2, 1]])
numeric_data = pd.DataFrame(data, columns=["a", "b", "c", "d"])
categorical_data = pd.DataFrame({"e": [0, 0, 1, 1], "f": [True, False, True, True]})
target_column = "e"


def test_chi_square_test():
    """Test that returned statistics for independence are correct."""
    output_statistics = chi_square_test(data=categorical_data, target_column=target_column, columns=("f"))
    np.testing.assert_array_equal((output_statistics["f"]), (0.0, 1.0, 1))


def test_normality_test():
    """Test that returned statistics for normality are correct."""
    output_statistics = normality_test(data=numeric_data)
    np.testing.assert_array_almost_equal(output_statistics["a"], (0.72863, 0.02386), decimal=5)


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
    output_matrix = correlation_matrix(data=numeric_data)
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
    output_matrix = covariance_matrix(data=numeric_data)
    np.testing.assert_array_almost_equal(output_matrix, expected_covariance_matrix)


def test_empty_df():
    """Test that empty DataFrame raises the correct exception."""
    empty_df = pd.DataFrame()
    with pytest.raises(exceptions.EmptyDataFrameException):
        normality_test(data=empty_df)


def test_invalid_columns():
    """Test that invalid column name in raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        chi_square_test(data=categorical_data, target_column=target_column, columns=["f", "x"])


def test_invalid_target_column():
    """Test that invalid target column raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        chi_square_test(data=categorical_data, target_column="invalid_column")


def test_invalid_correlation_method():
    """Test that invalid correlation method raises the correct exception."""
    with pytest.raises(BeartypeCallHintParamViolation):
        correlation_matrix(data=numeric_data, correlation_method="invalid_method")


def test_min_periods_with_kendall():
    """Test that min_periods with correlation_method 'kendall' raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        correlation_matrix(data=numeric_data, correlation_method="kendall", min_periods=1)


def test_invalid_ddof():
    """Test that invalid delta degrees of freedom raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        covariance_matrix(data=numeric_data, delta_degrees_of_freedom=-1)
