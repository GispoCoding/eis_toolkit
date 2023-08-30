import numpy as np
import pandas as pd
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from eis_toolkit import exceptions
from eis_toolkit.statistical_analyses.statistical_testing import statistical_tests

data = np.array([[0, 1, 2, 1], [2, 0, 1, 2], [2, 1, 0, 2], [0, 1, 2, 1]])
df = pd.DataFrame(data, columns=["a", "b", "c", "d"])
df["e"] = [0, 0, 1, 1]
df["f"] = [True, False, True, True]
target_column = "e"
categorical_variables = ["e", "f"]


def test_output():
    """Test that returned statistics are correct."""
    output = statistical_tests(data=df, target_column=target_column, categorical_variables=categorical_variables)
    expected_correlation_matrix = np.array(
        [
            [1.000000, -0.577350, -0.904534, 1.000000],
            [-0.577350, 1.000000, 0.174078, -0.577350],
            [-0.904534, 0.174078, 1.000000, -0.904534],
            [1.000000, -0.577350, -0.904534, 1.000000],
        ]
    )
    expected_covariance_matrix = np.array(
        [
            [1.333333, -0.333333, -1.000000, 0.666667],
            [-0.333333, 0.250000, 0.083333, -0.166667],
            [-1.000000, 0.083333, 0.916667, -0.500000],
            [0.666667, -0.166667, -0.500000, 0.333333],
        ]
    )
    np.testing.assert_array_almost_equal(output["correlation matrix"], expected_correlation_matrix)
    np.testing.assert_array_almost_equal(output["covariance matrix"], expected_covariance_matrix)
    np.testing.assert_array_almost_equal(output["normality"]["a"]["shapiro"], (0.72863, 0.02386), decimal=5)
    np.testing.assert_almost_equal(output["normality"]["a"]["anderson"][0], 0.576024, decimal=5)
    np.testing.assert_array_equal(output["normality"]["a"]["anderson"][1], [1.317, 1.499, 1.799, 2.098, 2.496])
    np.testing.assert_array_equal(
        (
            output["f"]["chi-square"],
            output["f"]["p-value"],
            output["f"]["degrees of freedom"],
        ),
        (0.0, 1.0, 1),
    )


def test_empty_df():
    """Test that empty DataFrame raises the correct exception."""
    empty_df = pd.DataFrame()
    with pytest.raises(exceptions.EmptyDataFrameException):
        statistical_tests(empty_df, target_column=target_column, categorical_variables=categorical_variables)


def test_invalid_target_column():
    """Test that invalid target column raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        statistical_tests(data=df, target_column="invalid_column", categorical_variables=categorical_variables)


def test_invalid_categorical_variable():
    """Test that invalid column name in categorical_variables raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        statistical_tests(data=df, target_column=target_column, categorical_variables=["e", "f", "x"])


def test_invalid_correlation_method():
    """Test that invalid correlation method raises the correct exception."""
    with pytest.raises(BeartypeCallHintParamViolation):
        statistical_tests(
            data=df,
            target_column=target_column,
            categorical_variables=categorical_variables,
            correlation_method="invalid_method",
        )


def test_min_periods_with_kendall():
    """Test that function call with min_periods and correlation_method 'kendall' raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        statistical_tests(
            data=df,
            target_column=target_column,
            categorical_variables=categorical_variables,
            correlation_method="kendall",
            min_periods=1,
        )


def test_invalid_ddof():
    """Test that invalid delta degrees of freedom raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        statistical_tests(
            data=df,
            target_column=target_column,
            categorical_variables=categorical_variables,
            delta_degrees_of_freedom=-1,
        )
