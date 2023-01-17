
from typing import Optional, Tuple
import pandas as pd
from sklearn.preprocessing import OneHotEncoder      # see www: sklearn OneHotEncod
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _separation(
    Xdf: pd.DataFrame,
    fields: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:            # Xvdf, Xcdf, ydf, 

    ### Target dataframe
    name = {i for i in fields if fields[i]=='t'}
    ydf = Xdf[list(name)]       #ydf = Xdf.loc[:,name]
    #Xdf.drop(name, axis=1, inplace=True)

    ### Values dataframe
    name ={i for i in fields if fields[i] in ('v','b')}
    Xvdf = Xdf[list(name)]       #ydf = Xdf.loc[:,name]
    #Xdf.drop(name, axis=1, inplace=True)

    ### Values dataframe
    name = {i for i in fields if fields[i] == 'c'}
    Xcdf = Xdf[list(name)]       #ydf = Xdf.loc[:,name]
    #Xdf.drop(name, axis=1, inplace=True)    

    return Xvdf, Xcdf, ydf

# *******************************
def separation(  # type: ignore[no-any-unimported]
    Xdf: pd.DataFrame,
    fields: dict
) -> Tuple[pd.DataFrame,pd.DataFrame, pd.DataFrame]:

    """
    Separates the target column to a separate dataframe.
    All categorical columns will be separated from all other features.

    Args:
        Xdf (pandas DataFrame): including target column ('t')
        fields (dictionary): column type for each column

    Returns:
        pandas DataFrame: value-sample  (Xvdf)
        pandas DataFrame: categorical features (Xcdf)
        pandas DataFrame: target (ydf)
    """

    Xvdf, Xcdf, ydf = _separation(Xdf,fields)

    return Xvdf, Xcdf, ydf


