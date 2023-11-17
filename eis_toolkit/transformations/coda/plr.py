import numpy as np
import pandas as pd
from beartype import beartype

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
    """TODO: docstring."""

    dfc = df.copy()

    if column not in dfc.columns:
        raise InvalidColumnException()

    idx = dfc.columns.index(column)

    if idx == len(dfc.columns) - 1:
        raise InvalidColumnException()

    # The denominator is a subcomposition of all the parts "to the right" of the column:
    # columns = [col for col in df.columns]
    # subcomposition = [columns[i] for i in len(columns) if i > idx]

    # TODO: finish implementation

    dfc = np.log()

    return np.log(df)


@beartype
def single_PLR_transform(df: pd.DataFrame):
    """
    Perform a pivot logratio transformation on the selected columns.

    Pivot logratio is a special case of ILR, where the numerator in the ratio is always a single
    part and the denominator all of the parts to the right in the ordered list of parts.

    Column order matters.

    TODO: Args, Returns, Raises
    """
    return
