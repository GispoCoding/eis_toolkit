
from typing import Tuple
import pandas as pd
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _unification(
    Xvdf: pd.DataFrame,
    Xcdf: pd.DataFrame,
) -> Tuple[pd.DataFrame]:

    return Xvdf.join(Xcdf)

# *******************************
def unification(
    Xvdf: pd.DataFrame,
    Xcdf: pd.DataFrame,
) -> Tuple[pd.DataFrame]:

    """
    unifies the to dataframes.

    Args:
        Xvdf (pandas DataFrame): Dataframe (with all value und binary fields)
        Xvdf (pandas DataFrame): Dataframe (with all categoriesed fields)

    Returns:
         pandas DataFrame (Xdf)
    """

    return _unification(Xvdf,Xcdf)


