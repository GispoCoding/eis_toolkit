import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Literal, Optional, Sequence, Tuple, Union
from scipy.stats import chi2_contingency, shapiro

from eis_toolkit import exceptions
from eis_toolkit.utilities.checks.dataframe import check_columns_numeric, check_columns_valid, check_empty_dataframe


@beartype
def chi_square_test(data: pd.DataFrame, target_column: str, columns: Optional[Sequence[str]] = None) -> dict:
    """Compute Chi-square test for independence on the input data.

    It is assumed that the variables in the input data are independent and that they are categorical, i.e. strings,
    booleans or integers, but not floats.

    Args:
        data: Dataframe containing the input data
        target_column: Variable against which independence of other variables is tested.
        columns: Variables that are tested against the variable in target_column. If None, every column is used.

    Raises:
        EmptyDataFrameException: The input Dataframe is empty.
        InvalidParameterValueException: The target_column is not in input Dataframe or invalid column is provided.

    Returns:
        Test statistics for each variable (except target_column).
    """
    if check_empty_dataframe(data):
        raise exceptions.EmptyDataFrameException("The input Dataframe is empty.")

    if not check_columns_valid(data, [target_column]):
        raise exceptions.InvalidParameterValueException("Target column not found in the Dataframe.")

    if columns is not None:
        invalid_columns = [column for column in columns if column not in data.columns]
        if any(invalid_columns):
            raise exceptions.InvalidParameterValueException(
                f"The following variables are not in the dataframe: {invalid_columns}"
            )
    else:
        columns = data.columns

    statistics = {}
    for column in columns:
        if column != target_column:
            contingency_table = pd.crosstab(data[target_column], data[column])
            chi_square, p_value, degrees_of_freedom, _ = chi2_contingency(contingency_table)
            statistics[column] = (chi_square, p_value, degrees_of_freedom)

    return statistics


@beartype
def normality_test(
    data: Union[pd.DataFrame, np.ndarray], columns: Optional[Sequence[str]] = None
) -> Union[dict, Tuple]:
    """Compute Shapiro-Wilk test for normality on the input data.

    Args:
        data: Dataframe or numpy array containing the input data.
        columns: Optional Columns to be used for testing.

    Returns:
        Test statistics for each variable.

    Raises:
        EmptyDataException: The input data is empty.
        NonNumericDataException: Selected data or columns contains non-numeric data.
        InvalidParameterValueException: Input column(s) are not in the data.
        ExceedingSampleSizeException: Input data exceeds the maximum of 5000 samples.
    """
    statistics = {}
    if isinstance(data, pd.DataFrame):
        if check_empty_dataframe(data):
            raise exceptions.EmptyDataException("The input Dataframe is empty.")

        if columns is not None:
            invalid_columns = [column for column in columns if column not in data.columns]
            if any(invalid_columns):
                raise exceptions.InvalidParameterValueException(
                    f"The following variables are not in the data: {invalid_columns}"
                )
            if not check_columns_numeric(data, columns):
                raise exceptions.NonNumericDataException("The selected columns contain non-numeric data.")
        else:
            if not check_columns_numeric(data, data.columns):
                raise exceptions.NonNumericDataException("The input data contain non-numeric data.")
            columns = data.columns

        for column in columns:
            if len(data[column]) > 5000:
                raise exceptions.SampleSizeExceededException(
                    f"Sample size for '{column}' exceeds the limit of 5000 samples."
                )
            statistic, p_value = shapiro(data[column])
            statistics[column] = (statistic, p_value)

    else:
        if data.size == 0:
            raise exceptions.EmptyDataException("The input numpy array is empty.")
        if not np.issubdtype(data.dtype, np.number):
            raise exceptions.NonNumericDataException("The input data contain non-numeric data.")
        if len(data) > 5000:
            raise exceptions.SampleSizeExceededException("Sample size exceeds the limit of 5000 samples.")

        flattened_data = data.flatten()
        statistic, p_value = shapiro(flattened_data)
        statistics = (statistic, p_value)

    return statistics


@beartype
def correlation_matrix(
    data: pd.DataFrame,
    correlation_method: Literal["pearson", "kendall", "spearman"] = "pearson",
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """Compute correlation matrix on the input data.

    It is assumed that the data is numeric, i.e. integers or floats.

    Args:
        data: Dataframe containing the input data.
        correlation_method: 'pearson', 'kendall', or 'spearman'. Defaults to 'pearson'.
        min_periods: Minimum number of observations required per pair of columns to have valid result. Optional.

    Raises:
        EmptyDataFrameException: The input Dataframe is empty.
        InvalidParameterValueException: min_periods argument is used with method 'kendall'.

    Returns:
        Dataframe containing the correlation matrix
    """
    if check_empty_dataframe(data):
        raise exceptions.EmptyDataFrameException("The input Dataframe is empty.")

    if correlation_method == "kendall" and min_periods is not None:
        raise exceptions.InvalidParameterValueException(
            "The argument min_periods is available only with correlation methods 'pearson' and 'spearman'."
        )

    matrix = data.corr(method=correlation_method, min_periods=min_periods, numeric_only=True)

    return matrix


@beartype
def covariance_matrix(
    data: pd.DataFrame, min_periods: Optional[int] = None, delta_degrees_of_freedom: int = 1
) -> pd.DataFrame:
    """Compute covariance matrix on the input data.

    It is assumed that the data is numeric, i.e. integers or floats.

    Args:
        data: Dataframe containing the input data.
        min_periods: Minimum number of observations required per pair of columns to have valid result. Optional.
        delta_degrees_of_freedom: Delta degrees of freedom used for computing covariance matrix. Defaults to 1.

    Raises:
        EmptyDataFrameException: The input Dataframe is empty.
        InvalidParameterValueException: Provided value for delta_degrees_of_freedom is negative.

    Returns:
        Dataframe containing the covariance matrix
    """
    if check_empty_dataframe(data):
        raise exceptions.EmptyDataFrameException("The input Dataframe is empty.")

    if delta_degrees_of_freedom < 0:
        raise exceptions.InvalidParameterValueException("Delta degrees of freedom must be non-negative.")

    matrix = data.cov(min_periods=min_periods, ddof=delta_degrees_of_freedom)

    return matrix
