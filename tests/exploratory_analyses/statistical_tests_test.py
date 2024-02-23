import numpy as np
import pandas as pd
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from eis_toolkit.exceptions import InvalidParameterValueException, NonNumericDataException
from eis_toolkit.exploratory_analyses.statistical_tests import chi_square_test, correlation_matrix, covariance_matrix

data = np.array([[0, 1, 2, 1], [2, 0, 1, 2], [2, 1, 0, 2], [0, 1, 2, 1]])
missing_data = np.array([[0, 1, 2, 1], [2, 0, np.nan, 2], [2, 1, 0, 2], [0, 1, 2, 1]])
non_numeric_data = np.array([[0, 1, 2, 1], ["a", "b", "c", "d"], [3, 2, 1, 0], ["c", "d", "b", "a"]])
numeric_data = pd.DataFrame(data, columns=["a", "b", "c", "d"])
non_numeric_df = pd.DataFrame(non_numeric_data, columns=["a", "b", "c", "d"])
missing_values_df = pd.DataFrame(missing_data, columns=["a", "b", "c", "d"])
categorical_data = pd.DataFrame({"e": [0, 0, 1, 1], "f": [True, False, True, True]})
target_column = "e"
np.random.seed(42)
large_data = np.random.normal(size=5001)
large_df = pd.DataFrame(large_data, columns=["a"])


def test_chi_square_test():
    """Test that returned statistics for independence are correct."""
    output_statistics = chi_square_test(data=categorical_data, target_column=target_column, columns=["f"])
    np.testing.assert_array_equal((output_statistics["f"]), (0.0, 1.0, 1))


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
    output_matrix = correlation_matrix(data=missing_values_df)
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
    output_matrix = correlation_matrix(data=numeric_data)
    np.testing.assert_array_almost_equal(output_matrix, expected_correlation_matrix)


def test_correlation_matrix_non_numeric():
    """Test that returned correlation matrix is correct."""
    with pytest.raises(NonNumericDataException):
        correlation_matrix(data=non_numeric_df)


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
    output_matrix = covariance_matrix(data=missing_values_df)
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


def test_covariance_matrix_negative_min_periods():
    """Test that negative min_periods value raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        covariance_matrix(data=numeric_data, min_periods=-1)


def test_invalid_target_column():
    """Test that invalid target column raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        chi_square_test(data=categorical_data, target_column="invalid_column")


def test_invalid_correlation_method():
    """Test that invalid correlation method raises the correct exception."""
    with pytest.raises(BeartypeCallHintParamViolation):
        correlation_matrix(data=numeric_data, correlation_method="invalid_method")


def test_min_periods_with_kendall():
    """Test that min_periods with correlation_method 'kendall' raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        correlation_matrix(data=numeric_data, correlation_method="kendall", min_periods=1)


def test_invalid_ddof():
    """Test that invalid delta degrees of freedom raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        covariance_matrix(data=numeric_data, delta_degrees_of_freedom=-1)
