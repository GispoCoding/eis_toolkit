import numpy as np
import pandas as pd
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from sklearn.datasets import load_iris

from eis_toolkit import exceptions
from eis_toolkit.statistical_analyses.statistical_testing import statistical_tests

iris_data = load_iris(as_frame=True)

# Create contingency table based on petal length and species variables
contingency_table = pd.crosstab(
    iris_data["target"], pd.cut(iris_data["data"]["petal length (cm)"], bins=[0, 3, 4, 5, 6, np.inf])
)


def test_output():
    """Test that returned statistics are correct."""
    output_numerical = statistical_tests(iris_data["data"])
    output_categorical = statistical_tests(contingency_table, data_type="categorical")
    np.testing.assert_array_almost_equal(
        output_numerical["correlation matrix"], np.corrcoef(iris_data["data"], rowvar=False)
    )
    np.testing.assert_array_almost_equal(output_numerical["covariance matrix"], np.cov(iris_data["data"], rowvar=False))
    np.testing.assert_array_almost_equal(
        output_numerical["normality"]["shapiro"]["petal length (cm)"], (0.876269, 7.412652e-10)
    )
    np.testing.assert_almost_equal(output_numerical["normality"]["anderson"]["petal width (cm)"][0], 5.105662)
    np.testing.assert_array_equal(
        output_numerical["normality"]["anderson"]["petal width (cm)"][1], [0.562, 0.64, 0.767, 0.895, 1.065]
    )
    np.testing.assert_array_almost_equal(
        (output_categorical["chi-square"], output_categorical["p-value"], output_categorical["degrees of freedom"]),
        (245.870894, 1.292214e-48, 8),
    )


def test_empty_df():
    """Test that empty DataFrame raises the correct exception."""
    empty_df = pd.DataFrame()
    with pytest.raises(exceptions.EmptyDataFrameException):
        statistical_tests(empty_df)


def test_invalid_data_type():
    """Test that invalid data type raises the correct exception."""
    with pytest.raises(BeartypeCallHintParamViolation):
        statistical_tests(data=iris_data["data"], data_type="invalid_type")


def test_invalid_method():
    """Test that invalid method raises the correct exception."""
    with pytest.raises(BeartypeCallHintParamViolation):
        statistical_tests(data=iris_data["data"], method="invalid_method")


def test_invalid_ddof():
    """Test that invalid delta degrees of freedom raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        statistical_tests(data=iris_data["data"], delta_degrees_of_freedom=-1)
