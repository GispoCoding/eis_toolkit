import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Optional, Sequence

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidParameterValueException, NonNumericDataException
from eis_toolkit.utilities.checks.dataframe import check_empty_dataframe


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
