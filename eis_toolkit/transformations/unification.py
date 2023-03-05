
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
       Unifies two dataframes.
    Args:
        Xvdf (pandas DataFrame): Dataframe (with all value und binary fields)
        Xcdf (pandas DataFrame): Dataframe (with all categoriesed fields)
    Returns:
         pandas DataFrame (Xdf)
    """
    
    # Argument evaluation
    fl = []
    if not (isinstance(Xcdf,pd.DataFrame)):
        fl.append('argument Xvdf is not a DataFrame')
    if not (isinstance(Xvdf,pd.DataFrame)):
        fl.append('argument Xcdf is not a DataFrame')
    if len(fl) > 0:
        raise InvalidParameterValueException ('***  function unification: ' + fl[0])
    if Xvdf.shape[0] != Xcdf.shape[0]:
        raise InvalidParameterValueException ('***  function unification: Xvdf and Xcdf have not the same number of rows')
    
    return _unification(Xvdf,Xcdf)

