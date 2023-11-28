import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Sequence

from eis_toolkit.exceptions import InvalidColumnIndexException
from eis_toolkit.utilities.aitchison_geometry import _closure
from eis_toolkit.utilities.checks.compositional import check_compositional
from eis_toolkit.utilities.checks.dataframe import check_column_index_in_dataframe
from eis_toolkit.utilities.miscellaneous import rename_columns_by_pattern


@beartype
def _alr_transform(df: pd.DataFrame, columns: Sequence[str], denominator_column: str) -> pd.DataFrame:

    ratios = df[columns].div(df[denominator_column], axis=0)
    return np.log(ratios)


@beartype
@check_compositional
def alr_transform(df: pd.DataFrame, idx: int = -1, keep_redundant_column: bool = False) -> pd.DataFrame:
    """
    Perform an additive logratio transformation on the data.

    Args:
        df: A dataframe of compositional data.
        idx: The integer position based index of the column of the dataframe to be used as denominator.
            If not provided, the last column will be used.
        keep_redundant_column: Whether to include the denominator column in the result. If True, the returned
            dataframe retains its original shape.

    Returns:
        A new dataframe containing the ALR transformed data.

    Raises:
        InvalidColumnIndexException: The input index for the denominator column is out of bounds.
        See check_compositional for other exceptions.
    """

    if not check_column_index_in_dataframe(df, idx):
        raise InvalidColumnIndexException("Denominator column index out of bounds.")

    denominator_column = df.columns[idx]
    columns = [col for col in df.columns]

    if not keep_redundant_column and denominator_column in columns:
        columns.remove(denominator_column)

    return rename_columns_by_pattern(_alr_transform(df, columns, denominator_column))


def inverse_alr(df: pd.DataFrame, column: str, scale: float = 1.0, idx: int = -1) -> pd.DataFrame:
    """
    Perform the inverse transformation for a set of ALR transformed data.

    Args:
        df: A dataframe of ALR transformed compositional data.
        column: The name of the redundant column.
        scale: The value to which each composition should be normalized. Eg., if the composition is expressed
            as percentages, scale=100.
        idx: Placement of the redundant column in the resulting dataframe.

    Returns:
        A dataframe containing the inverse transformed data.
    """
    dfc = df.copy()

    if column not in dfc.columns.values:
        # Add the denominator column
        dfc[column] = 0.0

    idx = dfc.columns.get_loc(column) if idx is None else idx
    idx = idx % len(dfc.columns)

    # Rearrange columns if necessary
    if idx != len(dfc.columns) - 1:
        original_columns = df.columns.tolist()
        length = len(original_columns)
        rearranged = original_columns[0:idx] + [column] + original_columns[idx:length]
        dfc = dfc[rearranged]

    dfc = _closure(np.exp(dfc), np.float64(scale))

    return dfc
