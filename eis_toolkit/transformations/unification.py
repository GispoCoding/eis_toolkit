import pandas as pd
from beartype import beartype

from eis_toolkit.exceptions import InvalidParameterValueException


# *******************************
@beartype
def _unification(
    Xvdf: pd.DataFrame,
    Xcdf: pd.DataFrame,
) -> pd.DataFrame:

    return Xvdf.join(Xcdf)


# *******************************
@beartype
def unification(
    Xvdf: pd.DataFrame,
    Xcdf: pd.DataFrame,
) -> pd.DataFrame:

    """
       Unifies two dataframes.
    Args:
        Xvdf: Dataframe (with all value und binary fields)
        Xcdf: Dataframe (with all categoriesed fields)
    Returns:
         pandas DataFrame cntaining all the columns.
    """

    # Args
    if Xvdf.shape[0] != Xcdf.shape[0]:
        raise InvalidParameterValueException("Xvdf and Xcdf have not the same number of rows")

    return _unification(Xvdf, Xcdf)
