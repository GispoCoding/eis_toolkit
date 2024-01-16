import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Sequence
from scipy.stats import gmean

from eis_toolkit.exceptions import InvalidColumnException, InvalidCompositionException, InvalidParameterValueException
from eis_toolkit.utilities.checks.compositional import check_in_simplex_sample_space
from eis_toolkit.utilities.checks.dataframe import check_columns_valid
from eis_toolkit.utilities.checks.parameter import check_lists_overlap, check_numeric_value_sign


@beartype
def _calculate_ilr_scaling_factor(c1: int, c2: int) -> np.float64:
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


@beartype
def _geometric_mean_logratio(
    row: pd.Series, subcomposition_1: Sequence[str], subcomposition_2: Sequence[str]
) -> np.float64:

    numerator = gmean(row[subcomposition_1])
    denominator = gmean(row[subcomposition_2])
    return np.log(numerator / denominator)


@beartype
def _single_ilr_transform(
    df: pd.DataFrame, subcomposition_1: Sequence[str], subcomposition_2: Sequence[str]
) -> pd.Series:

    dfc = df.copy()

    c1 = len(subcomposition_1)
    c2 = len(subcomposition_2)

    # A Series to hold the transformed rows
    ilr_values = pd.Series([0.0] * df.shape[0])

    for idx, row in dfc.iterrows():
        ilr_values[idx] = _geometric_mean_logratio(row, subcomposition_1, subcomposition_2)

    ilr_values = _calculate_ilr_scaling_factor(c1, c2) * ilr_values

    return ilr_values


@beartype
def single_ilr_transform(
    df: pd.DataFrame, subcomposition_1: Sequence[str], subcomposition_2: Sequence[str]
) -> pd.Series:
    """
    Perform a single isometric logratio transformation on the provided subcompositions.

    Returns ILR balances. Column order matters.

    Args:
        df: A dataframe of shape [N, D] of compositional data.
        subcomposition_1: Names of the columns in the numerator part of the ratio.
        subcomposition_2: Names of the columns in the denominator part of the ratio.

    Returns:
        A series of length N containing the transforms.

    Raises:
        InvalidColumnException: One or more subcomposition columns are not found in the input dataframe.
        InvalidCompositionException: Data is not normalized to the expected value or
            one or more columns are found in both subcompositions.
        InvalidParameterValueException: At least one subcomposition provided was empty.
        NumericValueSignException: Data contains zeros or negative values.
    """
    check_in_simplex_sample_space(df)

    if not (subcomposition_1 and subcomposition_2):
        raise InvalidParameterValueException("A subcomposition should contain at least one column.")

    if not (check_columns_valid(df, subcomposition_1) and check_columns_valid(df, subcomposition_2)):
        raise InvalidColumnException("Not all of the input columns were found in the input dataframe.")

    if check_lists_overlap(subcomposition_1, subcomposition_2):
        raise InvalidCompositionException("The subcompositions overlap.")

    return _single_ilr_transform(df, subcomposition_1, subcomposition_2)
