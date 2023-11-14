import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Optional, Sequence

from eis_toolkit.exceptions import InvalidColumnException
from eis_toolkit.utilities.checks.dataframe import check_columns_valid, check_dataframe_contains_zeros


@beartype
def _ILR_transform(df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Perform an isometric logratio transformation on the selected columns.

    Returns ILR balances Column order matters.

    TODO: Args, Returns, Raises
    """
    if columns is not None:
        if check_columns_valid(df, columns) is False:
            raise InvalidColumnException("Not all of the given columns were found in the input DataFrame.")

    if check_dataframe_contains_zeros(df):
        raise InvalidColumnException("The dataframe contains one or more zeros.")

    # TODO: implement

    return np.log(df)


@beartype
def _PLR_transform(df: pd.DataFrame):
    """
    Perform a pivot logratio transformation on the selected columns.

    Pivot logratio is a special case of ILR, where the numerator in the ratio is always a single
    part and the denominator all of the parts to the right in the ordered list of parts.

    Column order matters.

    TODO: Args, Returns, Raises
    """
    return np.log(df)
