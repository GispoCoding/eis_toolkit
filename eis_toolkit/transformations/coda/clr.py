import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Tuple
from scipy.stats import gmean

from eis_toolkit.utilities.aitchison_geometry import _closure
from eis_toolkit.utilities.checks.coda import check_compositional


@beartype
def _centered_ratio(row: pd.Series) -> pd.Series:

    return row / gmean(row)


@beartype
def _clr_transform(df: pd.DataFrame) -> pd.DataFrame:

    dfc = df.copy()
    dfc = dfc.apply(_centered_ratio, axis=1)

    return np.log(dfc)


@beartype
@check_compositional
def clr_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform a centered logratio transformation on the data.

    Args:
        df: A dataframe of compositional data.

    Returns:
        A new dataframe containing the CLR transformed data.

    Raises:
        See check_compositional.
    """
    return _clr_transform(df)


def inverse_clr(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.float64]:
    """Perform the inverse transformation for a set of CLR transformed data."""

    return _closure(np.exp(df))
