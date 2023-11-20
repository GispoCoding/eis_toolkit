import numpy as np
import pandas as pd
from beartype import beartype
from scipy.stats import gmean

from eis_toolkit.exceptions import InvalidColumnException, InvalidParameterValueException
from eis_toolkit.utilities.checks.parameter import check_numeric_value_sign


@beartype
def _calculate_scaling_factor(c: int) -> np.float64:
    """
    Calculate the scaling factor for the PLR transform.

    Args:
        c: The cardinality of the remaining parts in the composition.

    Returns:
        The scaling factor used performing a single PLR transform for a composition.

    Raises:
        InvalidParameterValueException: The input value is zero or negative.
    """
    if not (check_numeric_value_sign(c)):
        raise InvalidParameterValueException("The input value must be a positive integer.")

    return np.sqrt(c / np.float64(1 + c))


@beartype
def _single_PLR_transform(df: pd.DataFrame, column: str) -> pd.Series:
    dfc = df.copy()
    idx = dfc.columns.get_loc(column)

    # The denominator is a subcomposition of all the parts "to the right" of the column:
    columns = [col for col in df.columns]
    subcomposition = [columns[i] for i in range(len(columns)) if i > idx]
    c = len(subcomposition)
    scaling_factor = _calculate_scaling_factor(c)

    # A series to hold the transformed rows
    plr_values = pd.Series([0.0] * df.shape[0])

    for idx, row in dfc.iterrows():
        plr_values[idx] = scaling_factor * np.log(row[column] / gmean(row[subcomposition]))

    return plr_values


@beartype
def single_PLR_transform(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Perform a pivot logratio transformation on the selected column.

    Pivot logratio is a special case of ILR, where the numerator in the ratio is always a single
    part and the denominator all of the parts to the right in the ordered list of parts.

    Column order matters.

    Args:
        df: A dataframe of shape [N, D] of compositional data.
        column: The name of the numerator column to use for the transformation.

    Returns:
        A series of length N containing the transforms.

    Raises:
        InvalidColumnException: The input column isn't found in the dataframe, or there are no columns
            to the right of the given column.
    """

    if column not in df.columns:
        raise InvalidColumnException()

    idx = df.columns.get_loc(column)

    if idx == len(df.columns) - 1:
        raise InvalidColumnException()

    return _single_PLR_transform(df, column)


@beartype
def _inverse_PLR():
    raise NotImplementedError()
