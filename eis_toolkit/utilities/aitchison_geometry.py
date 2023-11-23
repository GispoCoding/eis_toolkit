import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Optional, Sequence, Tuple

from eis_toolkit.utilities.checks.dataframe import check_dataframe_contains_only_positive_numbers


@beartype
def check_in_simplex_sample_space(df: pd.DataFrame, k: np.float64 = None) -> bool:
    """
    Check that the compositions represented by the data rows belong to a simplex sample space.

    Checks that each compositional data point belongs to the set of positive real numbers.
    Checks that each composition is normalized to the same value.

    Args:
        df: Dataframe to check.
        k: The expected sum of each row. If None, simply checks that the sum of each row is equal.

    Returns:
        True if values are valid and the sum of each row is k.
    """
    if not check_dataframe_contains_only_positive_numbers(df):
        return False

    df_sum = np.sum(df, axis=1)
    expected_sum = k if k is not None else df_sum.iloc[0]
    if len(df_sum[df_sum.iloc[:] != expected_sum]) != 0:
        return False

    return True


@beartype
def check_in_unit_simplex_sample_space(df: pd.DataFrame) -> bool:
    """
    Check that the compositions represented by the data rows belong to the unit simplex sample space.

    Checks that each compositional data point belongs to the set of positive real numbers.
    Checks that each composition is normalized to 1.

    Args:
        df: Dataframe to check.

    Returns:
        True if values are valid and the sum of each row is 1.
    """
    return check_in_simplex_sample_space(df, np.float64(1))


@beartype
def _scale(df: pd.DataFrame, scale: np.float64) -> pd.DataFrame:
    """TODO: docstring."""
    return scale * df


@beartype
def _normalize(
    row: pd.Series, columns: Optional[Sequence[str]] = None, sum: np.float64 = 1.0
) -> Tuple[pd.Series, np.float64]:
    """TODO: docstring."""
    if columns is None:
        scale = np.float64(np.sum(row)) / sum
        row = np.divide(row, scale)
    else:
        scale = np.float64(np.sum(row[columns])) / sum
        row[columns] = np.divide(row[columns], scale)
    return row, scale


@beartype
def _normalize_to_one(row: pd.Series, columns: Optional[Sequence[str]] = None) -> Tuple[pd.Series, np.float64]:
    """
    Normalize the series to one.

    Args:
        row: The series to normalize.

    Returns:
        A tuple containing a new series with the normalized values and the scale factor used.
    """
    if columns is None:
        scale = np.float64(np.sum(row))
        row = np.divide(row, scale)
    else:
        scale = np.float64(np.sum(row[columns]))
        row[columns] = np.divide(row[columns], scale)
    return row, scale


@beartype
def _closure(df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Perform the closure operation on the dataframe.

    Assumes the standard simplex, in which the sum of the components of each composition vector is 1.

    Args:
        df: A dataframe of shape (N, D) compositional data.
        columns: Names of the columns to normalize.

    Returns:
        A new dataframe of shape (N, D), in which the specified columns have been normalized to 1,
        and other columns retain the data they had.
        A series containing the scale factor used to normalize each row. Uses the same indexing as the dataframe rows.

    Raises:
        # TODO
    """

    # TODO: add check/requirement for df having to contain non-numeric column names

    columns = [col for col in df.columns] if columns is None else columns

    dfc = df.copy()
    scales = pd.Series(np.zeros((len(dfc),)))

    for idx, row in df.iterrows():
        row, scale = _normalize_to_one(row, columns)
        dfc.iloc[idx] = row
        scales.iloc[idx] = scale

    return dfc, scales


# TODO (below): operations in the Aitchison geometry/simplex

# (POSSIBLE TODO: perturbation operation function)

# (POSSIBLE TODO: powering operation function)

# (POSSIBLE TODO: inner product operation)
