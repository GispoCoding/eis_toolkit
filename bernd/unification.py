"""
two df to unity to one df
State an Dezember 12 2022
@author: torchala 
""" 
# Stand: fast fertig: 

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
        Xvdf (DataFrame): Dataframe (with all valu und binary fields)
        Xvdf (DataFrame): Dataframe (with all categoriesed fields)

    Returns:
         pd.dataframe (Xdf)
    """

    return _unification(Xvdf,Xcdf)


