"""
State an Dezember 08 2022
@author: torchala 
""" 
# ydf aus Xdf herauslösen

#  Fehlermöglichkeit: keine oder mehrere t-Spalten

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder      # see www: sklearn OneHotEncod
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _target_separation(
    Xdf: pd.DataFrame,
    fields: dict
) -> Tuple[pd.DataFrame,pd.DataFrame]:            # Xdf, ydf

    ### Target dataframe
    name = [i for i in fields if fields[i]=="t"]
    ydf = Xdf[name[0]]       #ydf = Xdf.loc[:,name]
    Xdf.drop(name, axis=1, inplace=True)
    
    return Xdf, ydf

# *******************************
def target_separation(  # type: ignore[no-any-unimported]
    Xdf: pd.DataFrame,
    fields: dict
) -> Tuple[pd.DataFrame,pd.DataFrame]:

    """
    separates the target column to a dataframe.

    Args:
        Xdf (DataFrame): including target column ('t')
        fields (dictionary): column type for each column

    Returns:
        sample pd.dataframe (X)
        target pd.dataframe (y)

    """

    Xdf, ydf = _target_separation(Xdf,fields)

    return Xdf, ydf


