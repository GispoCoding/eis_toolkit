import pandas as pd
from beartype import beartype
from beartype.typing import Literal, Optional, Sequence, Tuple
from scipy.stats import chi2_contingency, shapiro

from eis_toolkit import exceptions


@beartype
def check_empty_dataframe(data: pd.DataFrame):
    """Check if the input dataframe is empty.

    Args:
        data: Input DataFrame

    Raises:
        EmptyDataFrameException: The input DataFrame is empty.
    """
    if data.empty:
        raise exceptions.EmptyDataFrameException("The input DataFrame is empty.")


@beartype
def chi_square_test(data: pd.DataFrame, target_column: str) -> Sequence[Tuple[float, float, int]]:
    """Compute Chi-square test for independence on categorical data.

    Args:
        data: DataFrame containing the input data.
        target_column: Variable against which independence of other variables is tested.

    Raises:
        InvalidParameterValueException: The target_column is not in input DataFrame.

    Returns:
        Test statistics for each variable (except target_column).
    """
    check_empty_dataframe(data)

    if target_column not in data.columns:
        raise exceptions.InvalidParameterValueException("Target column not found in the DataFrame.")

    statistics = []
    for column in data.columns:
        if column != target_column:
            contingency_table = pd.crosstab(data[target_column], data[column])
            chi_square, p_value, degrees_of_freedom, _ = chi2_contingency(contingency_table)
            statistics.append((chi_square, p_value, degrees_of_freedom))

    return statistics


@beartype
def normality_test(data: pd.DataFrame) -> Sequence[Tuple[float, float]]:
    """Compute Shapiro-Wilk test for normality on numeric input data.

    Args:
        data: DataFrame containing the input data.

    Returns:
        Test statistics for each variable.
    """
    check_empty_dataframe(data)

    statistics = []
    for column in data.columns:
        statistic, p_value = shapiro(data[column])
        statistics.append((statistic, p_value))

    return statistics


@beartype
def correlation_matrix(
    data: pd.DataFrame,
    correlation_method: Literal["pearson", "kendall", "spearman"] = "pearson",
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """Compute correlation matrix on numeric input data.

    Args:
        data: DataFrame containing the input data.
        correlation_method: 'pearson', 'kendall', or 'spearman'. Defaults to 'pearson'.
        min_periods: Minimum number of observations required per pair of columns to have valid result. Optional.

    Raises:
        InvalidParameterValueException: min_periods argument is used with method 'kendall'.

    Returns:
        Correlation matrix
    """
    check_empty_dataframe(data)

    if correlation_method == "kendall" and min_periods is not None:
        raise exceptions.InvalidParameterValueException(
            "The argument min_periods is available only with correlation methods 'pearson' and 'spearman'."
        )

    return data.corr(method=correlation_method, min_periods=min_periods)


@beartype
def covariance_matrix(
    data: pd.DataFrame, min_periods: Optional[int] = None, delta_degrees_of_freedom: int = 1
) -> pd.DataFrame:
    """Compute covariance matrix on numeric input data.

    Args:
        data: DataFrame containing the input data.
        min_periods: Minimum number of observations required per pair of columns to have valid result. Optional.
        delta_degrees_of_freedom: Delta degrees of freedom used for computing covariance matrix. Defaults to 1.

    Raises:
        InvalidParameterValueException: Provided value for delta_degrees_of_freedom is negative.

    Returns:
        Covariance matrix
    """
    check_empty_dataframe(data)

    if delta_degrees_of_freedom < 0:
        raise exceptions.InvalidParameterValueException("Delta degrees of freedom must be non-negative.")

    return data.cov(min_periods=min_periods, ddof=delta_degrees_of_freedom)
