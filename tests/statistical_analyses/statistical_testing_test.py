import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from beartype.roar import BeartypeCallHintParamViolation

from eis_toolkit import exceptions
from eis_toolkit.statistical_analyses.statistical_testing import statistical_tests

tips_data = sns.load_dataset("tips")
target_column = "size"
numerical_columns = tips_data.select_dtypes(include=["float"])


def test_output():
    """Test that returned statistics are correct."""
    output = statistical_tests(tips_data, target_column=target_column)
    np.testing.assert_array_almost_equal(output["correlation matrix"], np.corrcoef(numerical_columns, rowvar=False))
    np.testing.assert_array_almost_equal(output["covariance matrix"], np.cov(numerical_columns, rowvar=False))
    np.testing.assert_array_almost_equal(output["normality"]["total_bill"]["shapiro"], (0.919719, 3.324453e-10))
    np.testing.assert_almost_equal(output["normality"]["total_bill"]["anderson"][0], 5.5207055)
    np.testing.assert_array_equal(output["normality"]["total_bill"]["anderson"][1], [0.567, 0.646, 0.775, 0.904, 1.075])
    np.testing.assert_array_almost_equal(
        (
            output["sex"]["chi-square"],
            output["sex"]["p-value"],
            output["sex"]["degrees of freedom"],
        ),
        (5.843737, 0.321722, 5),
    )


def test_empty_df():
    """Test that empty DataFrame raises the correct exception."""
    empty_df = pd.DataFrame()
    with pytest.raises(exceptions.EmptyDataFrameException):
        statistical_tests(empty_df, target_column=target_column)


def test_invalid_method():
    """Test that invalid method raises the correct exception."""
    with pytest.raises(BeartypeCallHintParamViolation):
        statistical_tests(data=tips_data, target_column=target_column, method="invalid_method")


def test_invalid_ddof():
    """Test that invalid delta degrees of freedom raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        statistical_tests(data=tips_data, target_column=target_column, delta_degrees_of_freedom=-1)
