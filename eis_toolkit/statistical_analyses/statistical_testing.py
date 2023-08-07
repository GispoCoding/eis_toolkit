import pandas as pd
from beartype import beartype
from beartype.typing import Literal, Optional
from scipy.stats import anderson, chi2_contingency, shapiro

from eis_toolkit import exceptions


def _statistical_tests(
    data: pd.DataFrame,
    data_type: Literal,
    target_column: Optional[str],
    method: Literal,
    min_periods: Optional[int],
    delta_degrees_of_freedom: int,
) -> dict:

    if data_type == "numerical":
        correlation_matrix = data.corr(method=method, min_periods=min_periods)
        covariance_matrix = data.cov(min_periods=min_periods, ddof=delta_degrees_of_freedom)

        normality_shapiro = {}
        normality_anderson = {}
        for column in data.columns:
            normality_shapiro[column] = shapiro(data[column])
            normality_anderson[column] = anderson(data[column], "norm")

        normality = {}
        normality["shapiro"] = normality_shapiro
        normality["anderson"] = normality_anderson

        statistics = {
            "correlation matrix": correlation_matrix,
            "covariance matrix": covariance_matrix,
            "normality": normality,
        }

    if data_type == "categorical":
        statistics = {}

        for column in data.columns:
            if column != target_column:
                contingency_table = pd.crosstab(data[target_column], data[column])
                chi_square, p_value, degree_of_freedom, _ = chi2_contingency(contingency_table)

                statistics[column] = {
                    "chi-square": chi_square,
                    "p-value": p_value,
                    "degrees of freedom": degree_of_freedom,
                }

    return statistics


@beartype
def statistical_tests(
    data: pd.DataFrame,
    data_type: Literal["numerical", "categorical"] = "numerical",
    target_column: Optional[str] = None,
    method: Literal["pearson", "kendall", "spearman"] = "pearson",
    min_periods: Optional[int] = None,
    delta_degrees_of_freedom: int = 1,
) -> dict:
    """Compute statistical tests on input data.

    Computes correlation and covariance matrices and normality statistics (Shapiro-Wilk and Anderson-Darling tests) for
    numerical data and independence statistic (Chi-square test) for categorical data.

    Args:
        data: DataFrame containing the input data. Contingency table if categorical data.
        data_type: 'numerical' or 'categorial'. Defaults to 'numerical'.
        target_column: Variable against which independence of other variables is tested.
        method: Correlation method: 'pearson', 'kendall' or 'spearman. Defaults to 'pearson'.
        min_periods: Minimum number of observations required per pair of columns to have valid result, optional.
        delta_degrees_of_freedom: Delta degrees of freedom used in computing covariance matrix. Defaults to 1.

    Raises:
        EmptyDataFrameException: The input DataFrame is empty.
        InvalidParameterValueException: Target column is not supplied when data_type is set 'categorical' or
            minimum number of observations per pair is not at least one nor None or
            delta degrees of freedom is negative.

    Returns:
        Dictionary containing computed statistics.
    """
    if data.empty:
        raise exceptions.EmptyDataFrameException("The input DataFrame is empty.")

    if data_type == "categorical" and target_column is None:
        raise exceptions.InvalidParameterValueException("Target column must be supplied with categorical data.")

    if min_periods is not None and min_periods < 1:
        raise exceptions.InvalidParameterValueException("Minimum number of observations per pair must be at least one.")

    if delta_degrees_of_freedom < 0:
        raise exceptions.InvalidParameterValueException("Delta degrees of freedom must be non-negative.")

    return _statistical_tests(data, data_type, target_column, method, min_periods, delta_degrees_of_freedom)
