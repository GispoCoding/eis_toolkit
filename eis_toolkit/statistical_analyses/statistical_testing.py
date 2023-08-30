import pandas as pd
from beartype import beartype
from beartype.typing import Literal, Optional, Sequence
from scipy.stats import anderson, chi2_contingency, shapiro

from eis_toolkit import exceptions


def _statistical_tests(
    data: pd.DataFrame,
    target_column: str,
    categorical_variables: Sequence,
    correlation_method: Literal,
    min_periods: Optional[int],
    delta_degrees_of_freedom: int,
) -> dict:

    statistics = {}
    normality = {}

    for column in data.columns:
        if column in categorical_variables:
            if column != target_column:
                contingency_table = pd.crosstab(data[target_column], data[column])
                chi_square, p_value, degree_of_freedom, _ = chi2_contingency(contingency_table)
                statistics[column] = {
                    "chi-square": chi_square,
                    "p-value": p_value,
                    "degrees of freedom": degree_of_freedom,
                }
        else:
            normality[column] = {"shapiro": shapiro(data[column]), "anderson": anderson(data[column], "norm")}

    if normality:
        statistics["normality"] = normality

    # Create subset of numerical variables only
    numerical_columns = data.loc[:, ~data.columns.isin(categorical_variables)]
    if not numerical_columns.empty:
        statistics["correlation matrix"] = numerical_columns.corr(method=correlation_method, min_periods=min_periods)
        statistics["covariance matrix"] = numerical_columns.cov(min_periods=min_periods, ddof=delta_degrees_of_freedom)

    return statistics


@beartype
def statistical_tests(
    data: pd.DataFrame,
    target_column: str,
    categorical_variables: Sequence,
    correlation_method: Literal["pearson", "kendall", "spearman"] = "pearson",
    min_periods: Optional[int] = None,
    delta_degrees_of_freedom: int = 1,
) -> dict:
    """Compute statistical tests on input data.

    Computes correlation and covariance matrices and normality statistics (Shapiro-Wilk and Anderson-Darling tests) for
    numerical variables and independence statistic (Chi-square test) for categorical variables.

    Args:
        data: DataFrame containing the input data.
        target_column: Variable against which independence of other variables is tested.
        categorical_variables: Variables that are considered categorical, i.e. not numerical.
        correlation_method: 'pearson', 'kendall' or 'spearman. Defaults to 'pearson'.
        min_periods: Minimum number of observations required per pair of columns to have valid result. Optional.
        delta_degrees_of_freedom: Delta degrees of freedom used in computing covariance matrix. Defaults to 1.

    Raises:
        EmptyDataFrameException: The input DataFrame is empty.
        InvalidParameterValueException: The target_column is not in input DataFrame or
            variable(s) in categorical_variables that do not exist in the DataFrame or
            min_periods argument is used with method 'kendall' or
            delta degrees of freedom is negative.

    Returns:
        Dictionary containing computed statistics.
    """
    if data.empty:
        raise exceptions.EmptyDataFrameException("The input DataFrame is empty.")

    if target_column not in data.columns:
        raise exceptions.InvalidParameterValueException("Target column not found in the DataFrame.")

    invalid_variables = [variable for variable in categorical_variables if variable not in data.columns]
    if any(invalid_variables):
        raise exceptions.InvalidParameterValueException(
            f"The following variables are not in the dataframe: {invalid_variables}"
        )

    if correlation_method == "kendall" and min_periods is not None:
        raise exceptions.InvalidParameterValueException(
            "The argument min_periods is available only with correlation methods 'pearson' and 'spearman'."
        )

    if delta_degrees_of_freedom < 0:
        raise exceptions.InvalidParameterValueException("Delta degrees of freedom must be non-negative.")

    return _statistical_tests(
        data, target_column, categorical_variables, correlation_method, min_periods, delta_degrees_of_freedom
    )
