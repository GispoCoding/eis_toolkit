import pandas as pd
from beartype import beartype
from beartype.typing import Literal, Optional
from scipy.stats import anderson, chi2_contingency, shapiro

from eis_toolkit import exceptions


def _statistical_tests(
    data: pd.DataFrame, target_column: str, method: Literal, min_periods: Optional[int], delta_degrees_of_freedom: int
) -> dict:

    statistics = {}
    normality = {}

    for column in data.columns:
        if data[column].dtype != float:  # Categorical variables
            if column != target_column:
                contingency_table = pd.crosstab(data[target_column], data[column])
                chi_square, p_value, degree_of_freedom, _ = chi2_contingency(contingency_table)
                statistics[column] = {
                    "chi-square": chi_square,
                    "p-value": p_value,
                    "degrees of freedom": degree_of_freedom,
                }
        else:  # Numerical variables
            normality[column] = {"shapiro": shapiro(data[column]), "anderson": anderson(data[column], "norm")}

    if normality:
        statistics["normality"] = normality

    # Create subset of numerical variables only
    numerical_columns = data.select_dtypes(include=["float"])
    if not numerical_columns.empty:
        statistics["correlation matrix"] = numerical_columns.corr(method=method, min_periods=min_periods)
        statistics["covariance matrix"] = numerical_columns.cov(min_periods=min_periods, ddof=delta_degrees_of_freedom)

    return statistics


@beartype
def statistical_tests(
    data: pd.DataFrame,
    target_column: str,
    method: Literal["pearson", "kendall", "spearman"] = "pearson",
    min_periods: Optional[int] = None,
    delta_degrees_of_freedom: int = 1,
) -> dict:
    """Compute statistical tests on input data.

    Computes correlation and covariance matrices and normality statistics (Shapiro-Wilk and Anderson-Darling tests) for
    numerical variables and independence statistic (Chi-square test) for categorical variables. NOTE: The function
    assumes all numerical variables are floats.

    Args:
        data: DataFrame containing the input data.
        target_column: Variable against which independence of other variables is tested.
        method: Correlation method: 'pearson', 'kendall' or 'spearman. Defaults to 'pearson'.
        min_periods: Minimum number of observations required per pair of columns to have valid result. Optional.
        delta_degrees_of_freedom: Delta degrees of freedom used in computing covariance matrix. Defaults to 1.

    Raises:
        EmptyDataFrameException: The input DataFrame is empty.
        InvalidParameterValueException: The target_column is not in input DataFrame or
            minimum number of observations per pair is not at least one nor None or
            delta degrees of freedom is negative.

    Returns:
        Dictionary containing computed statistics.
    """
    if data.empty:
        raise exceptions.EmptyDataFrameException("The input DataFrame is empty.")

    if target_column not in data.columns:
        raise exceptions.InvalidParameterValueException("Target column not found in the DataFrame.")

    if min_periods is not None and min_periods < 1:
        raise exceptions.InvalidParameterValueException("Minimum number of observations per pair must be at least one.")

    if delta_degrees_of_freedom < 0:
        raise exceptions.InvalidParameterValueException("Delta degrees of freedom must be non-negative.")

    return _statistical_tests(data, target_column, method, min_periods, delta_degrees_of_freedom)
