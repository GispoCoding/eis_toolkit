import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Dict, Literal, Optional, Sequence
from scipy.stats import chi2_contingency

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidParameterValueException, NonNumericDataException
from eis_toolkit.utilities.checks.dataframe import check_columns_valid, check_empty_dataframe


@beartype
def chi_square_test(
    data: pd.DataFrame, target_column: str, columns: Optional[Sequence[str]] = None
) -> Dict[str, Dict[str, float]]:
    """Perform a Chi-square test of independence between a target variable and one or more other variables.

    Input data should be categorical data. Continuous data or non-categorical data should be discretized or
    binned before using this function, as Chi-square tests are not applicable to continuous variables directly.

    The test assumes that the observed frequencies in each category are independent.

    Args:
        data: Dataframe containing the input data.
        target_column: Variable against which independence of other variables is tested.
        columns: Variables that are tested against the variable in target_column. If None, every column is used.

    Returns:
        Test statistics, p-value and degrees of freedom for each variable.

    Raises:
        EmptyDataFrameException: Input Dataframe is empty.
        InvalidParameterValueException: Invalid column is input.
    """
    if check_empty_dataframe(data):
        raise EmptyDataFrameException("The input Dataframe is empty.")

    if not check_columns_valid(data, [target_column]):
        raise InvalidParameterValueException("Target column not found in the Dataframe.")

    if columns:
        invalid_columns = [column for column in columns if column not in data.columns]
        if invalid_columns:
            raise InvalidParameterValueException(f"Invalid columns: {invalid_columns}")
    else:
        columns = [col for col in data.columns if col != target_column]

    statistics = {}
    for column in columns:
        contingency_table = pd.crosstab(data[target_column], data[column])
        chi_square, p_value, degrees_of_freedom, _ = chi2_contingency(contingency_table)
        statistics[column] = {"chi_square": chi_square, "p-value": p_value, "degrees_of_freedom": degrees_of_freedom}

    return statistics


@beartype
def correlation_matrix(
    data: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    correlation_method: Literal["pearson", "kendall", "spearman"] = "pearson",
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """Compute correlation matrix on the input data.

    It is assumed that the data is numeric, i.e. integers or floats. NaN values are excluded from the calculations.

    Args:
        data: Dataframe containing the input data.
        columns: Columns to include in the correlation matrix. If None, all numeric columns are used.
        correlation_method: 'pearson', 'kendall', or 'spearman'. Defaults to 'pearson'.
        min_periods: Minimum number of observations required per pair of columns to have valid result. Optional.

    Returns:
        Dataframe containing matrix representing the correlation coefficient \
            between the corresponding pair of variables.

    Raises:
        EmptyDataFrameException: The input Dataframe is empty.
        InvalidParameterValueException: min_periods argument is used with method 'kendall'.
        NonNumericDataException: The selected columns contain non-numeric data.
    """
    if check_empty_dataframe(data):
        raise EmptyDataFrameException("The input Dataframe is empty.")

    if columns:
        invalid_columns = [column for column in columns if column not in data.columns]
        if invalid_columns:
            raise InvalidParameterValueException(f"Invalid columns: {invalid_columns}")
        data_subset = data[columns]
    else:
        data_subset = data.select_dtypes(include=np.number)

    if not all(data_subset.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        raise NonNumericDataException("The input data contain non-numeric data.")

    if correlation_method == "kendall" and min_periods is not None:
        raise InvalidParameterValueException(
            "The argument min_periods is available only with correlation methods 'pearson' and 'spearman'."
        )

    matrix = data_subset.corr(method=correlation_method, min_periods=min_periods, numeric_only=True)

    return matrix


@beartype
def covariance_matrix(
    data: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    min_periods: Optional[int] = None,
    delta_degrees_of_freedom: int = 1,
) -> pd.DataFrame:
    """Compute covariance matrix on the input data.

    It is assumed that the data is numeric, i.e. integers or floats. NaN values are excluded from the calculations.

    Args:
        data: Dataframe containing the input data.
        columns: Columns to include in the covariance matrix. If None, all numeric columns are used.
        min_periods: Minimum number of observations required per pair of columns to have valid result. Optional.
        delta_degrees_of_freedom: Delta degrees of freedom used for computing covariance matrix. Defaults to 1.

    Returns:
        Dataframe containing matrix representing the covariance between the corresponding pair of variables.

    Raises:
        EmptyDataFrameException: The input Dataframe is empty.
        InvalidParameterValueException: Provided value for delta_degrees_of_freedom or min_periods is negative.
        NonNumericDataException: The input data contain non-numeric data.
    """
    if check_empty_dataframe(data):
        raise EmptyDataFrameException("The input Dataframe is empty.")

    if columns:
        invalid_columns = [column for column in columns if column not in data.columns]
        if invalid_columns:
            raise InvalidParameterValueException(f"Invalid columns: {invalid_columns}")
        data_subset = data[columns]
    else:
        data_subset = data.select_dtypes(include=np.number)

    if not all(data_subset.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        raise NonNumericDataException("The input data contain non-numeric data.")

    if delta_degrees_of_freedom < 0:
        raise InvalidParameterValueException("Delta degrees of freedom must be non-negative.")

    if min_periods and min_periods < 0:
        raise InvalidParameterValueException("Min perioids must be non-negative.")

    matrix = data_subset.cov(min_periods=min_periods, ddof=delta_degrees_of_freedom)

    return matrix
