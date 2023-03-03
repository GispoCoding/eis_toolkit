
from typing import Optional, Tuple
import pandas as pd
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _all_separation(
    df: pd.DataFrame,
    fields: dict,
) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:            # Xvdf, Xcdf, ydf, ,

    ### Target dataframe
    name = {i for i in fields if fields[i]=='t'}
    ydf = df[list(name)]       #ydf = Xdf.loc[:,name]
    #Xdf.drop(name, axis=1, inplace=True)

    ### Values dataframe
    name ={i for i in fields if fields[i] in ('v','b')}
    Xvdf = df[list(name)]       #ydf = Xdf.loc[:,name]
    #Xdf.drop(name, axis=1, inplace=True)

    ### classes dataframe
    name = {i for i in fields if fields[i] == 'c'}
    Xcdf = df[list(name)]       #ydf = Xdf.loc[:,name]
    #Xdf.drop(name, axis=1, inplace=True)    

    ### identity-geometry dataframe
    name = {i for i in fields if fields[i] in ('i','g')}
    igdf = df[list(name)]       #ydf = Xdf.loc[:,name]
    #Xdf.drop(name, axis=1, inplace=True)   

    return Xvdf,Xcdf,ydf,igdf

# *******************************
def all_separation(  # type: ignore[no-any-unimported]
    df: pd.DataFrame,
    fields: dict
) -> Tuple[pd.DataFrame,pd.DataFrame, pd.DataFrame,pd.DataFrame]:

    """
        Separates the target column to a separate dataframe.
        All categorical columns (fields) will be separated from all other features.
    Args:
        df (pandas DataFrame): including target column ('t')
        fields (dictionary): column type for each column

    Returns:
        pandas DataFrame: value-sample  (Xvdf)
        pandas DataFrame: categorical columns (Xcdf)
        pandas DataFrame: target (ydf)
        pandas DataFrame: target (igdf)
    """

    # Argument evaluation
    fl = []
    if not (isinstance(df,pd.DataFrame)):
        fl.append('argument df is not a DataFrame')
    if not (isinstance(fields,dict)):
        fl.append('argument fields is not a dictionary') 
    if len(fl) > 0:
        raise InvalidParameterValueException ('***  function all_separation: ' + fl[0])
        
    if len(df.columns) == 0:
        raise InvalidParameterValueException ('***  function all_nodata_remove: DataFrame has no column')
    if len(df.index) == 0:
        raise InvalidParameterValueException ('***  function all_nodata_remove: DataFrame has no rows')
    if len(fields) == 0:
        raise InvalidParameterValueException ('***  function all_nodata_remove: Fields is empty')

    # call
    Xvdf,Xcdf,ydf,igdf = _all_separation(df,fields)

    return Xvdf,Xcdf,ydf,igdf
