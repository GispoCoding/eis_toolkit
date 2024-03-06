import pandas as pd
from beartype import beartype
from beartype.typing import Dict, Optional, Sequence
from scipy.stats import chi2_contingency

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidParameterValueException
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
