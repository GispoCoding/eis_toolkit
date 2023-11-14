import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Optional, Sequence, Tuple

from eis_toolkit.utilities.checks.dataframe import check_dataframe_contains_only_positive_numbers


@beartype
def check_composition_belongs_to_unit_simplex_sample_space(df: pd.DataFrame) -> bool:
    """
    Check that the compositions represented by the data rows belong to the unit simplex sample space.

    Check that each compositional data point belongs to the set of positive real numbers.
    Check that each composition is normalized to 1.

    #TODO: Args, Returns, Raises

    Returns:
        InvalidCompositionException: Data is not normalized to 1.
        NumericValueSignException: Data contains zeros or negative values.
    """
    if not check_dataframe_contains_only_positive_numbers(df):
        return False
        # raise NumericValueSignException("Data contains zeros or negative values.")

    # TODO: switch to checking that the sum is the same value for each column
    df_sum = np.sum(df, axis=1)
    if len(df_sum[df_sum.iloc[:] != 1]) != 0:
        return False
        # raise InvalidCompositionException("Not each composition is normalized to 1.")

    return True


@beartype
def _normalize_to_one(row: pd.Series, columns: Sequence[str]) -> Tuple[pd.Series, np.float64]:
    """
    Normalize the series to one.

    Args:
        row: The series to normalize.

    Returns:
        A tuple containing a new series with the normalized values and the scale factor used.
    """
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
    columns = df.columns if columns is None else columns

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
