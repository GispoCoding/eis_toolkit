import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Sequence
from scipy.stats import gmean

from eis_toolkit.exceptions import InvalidColumnException, InvalidParameterValueException
from eis_toolkit.utilities.checks.coda import check_compositional
from eis_toolkit.utilities.checks.dataframe import check_columns_valid, check_dataframe_contains_zeros
from eis_toolkit.utilities.checks.parameter import check_numeric_value_sign


@beartype
def _calculate_scaling_factor(c1: int, c2: int) -> np.float64:
    """
    Calculate the scaling factor for the ILR transform.

    Args:
        c1: The cardinality of the first subcomposition.
        c2: The cardinality of the second subcomposition.

    Returns:
        The scaling factor.

    Raises:
        InvalidParameterValueException: One or both of the input values are zero or negative.
    """
    if not (check_numeric_value_sign(c1) and check_numeric_value_sign(c2)):
        raise InvalidParameterValueException("Input values must both be positive integers.")

    return np.sqrt((c1 * c2) / np.float64(c1 + c2))


# TODO: name better
@beartype
def _logratio(row: pd.Series, subcomposition_1: Sequence[str], subcomposition_2: Sequence[str]) -> np.float64:
    """TODO: docstring."""
    numerator = gmean(row[subcomposition_1])
    denominator = gmean(row[subcomposition_2])
    return np.log(numerator / denominator)


@beartype
def _single_ILR_transform(
    df: pd.DataFrame, subcomposition_1: Sequence[str], subcomposition_2: Sequence[str]
) -> pd.Series:
    """TODO: docstring."""

    dfc = df.copy()

    c1 = len(subcomposition_1)
    c2 = len(subcomposition_2)

    # A Series to hold the transformed rows
    ilr_values = pd.Series([0.0] * df.shape[0])

    for idx, row in dfc.iterrows():
        ilr_values[idx] = _logratio(row, subcomposition_1, subcomposition_2)

    ilr_values = _calculate_scaling_factor(c1, c2) * ilr_values

    return ilr_values


@beartype
@check_compositional
def single_ILR_transform(
    df: pd.DataFrame, subcomposition_1: Sequence[str], subcomposition_2: Sequence[str]
) -> pd.Series:
    """
    Perform a single isometric logratio transformation on the provided subcompositions.

    Returns ILR balances.
    Column order matters.

    TODO: Args, Returns, Raises
    """

    # TODO: verify whether the subcompositions are allowed to have overlap

    if not (check_columns_valid(df, subcomposition_1) and check_columns_valid(df, subcomposition_2)):
        raise InvalidColumnException("Not all of the given columns were found in the input DataFrame.")

    # TODO: possibly only check the subcomposition columns
    if check_dataframe_contains_zeros(df):
        raise InvalidColumnException("The dataframe contains one or more zeros.")

    return _single_ILR_transform(df, subcomposition_1, subcomposition_2)


@beartype
def _ILR_inverse():
    raise NotImplementedError()
