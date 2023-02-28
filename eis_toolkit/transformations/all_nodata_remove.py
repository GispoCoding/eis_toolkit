
from typing import Tuple
import copy
import numpy as np 
import pandas as pd
#from all_separation import *
#from all_unification import *
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _all_nodata_remove(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame,pd.DataFrame]:     # das 2. df ist die neue nodata-mask (nur eine Spalte, ebenso viele reihen, wie die erste df)

    # Argument evaluation
    fl = []
    if not (isinstance(df,pd.DataFrame)):
        fl.append('argument df is not a DataFrame')
    if len(fl) > 0:
        raise InvalidParameterValueException ('***  function all_nodata_remove: ' + fl[0])

    if len(df.columns) == 0:
        raise InvalidParameterValueException ('***  function all_nodata_remove: DataFrame has no column')
    if len(df.index) == 0:
        raise InvalidParameterValueException ('***  function all_nodata_remove: DataFrame has no rows')
        
    # datatype to float32 (on this way float64 come to np.inf)
    df.loc[:,df.dtypes=='float64'] = df.loc[:,df.dtypes=='float64'].astype('float32')
    df.loc[:,df.dtypes=='int64'] = df.loc[:,df.dtypes=='int64'].astype('int32')
    df.replace([np.inf, -np.inf],np.nan,inplace=True)
    # for col in df.columns:
    #     if not is_numeric_dtype(df[col]):   # df[col].dtype != np.number:    # # empty strings cells will get numpy.nan
    #Xdf.replace(r'^\s*$',np.nan,regex=True,inplace=True)   # or: df.replace(r'\s+', np.nan, regex=True)

    # Xvdf, Xcdf, ydf = separation(Xdf = Xdf, fields = columns)
    # Xdfv = Xdfv.astype('float32')
    # Xdf.replace([np.inf, -np.inf], np.nan, inplace=True)
    #unification

    dfmask = df.isna()      # dfmsk will be needed in module "export_grid" after prediction to add nodata-values in y after prediction
    dfsum = pd.DataFrame(dfmask.sum(axis=1))
    dfsum[dfsum>0] = True
    dfsum[dfsum==0] = False
    #dfsum.replace({1:True,0:False},inplace = True)
    df.dropna(inplace = True)    # drops rows which contain missing values
    df = copy.deepcopy(df.reset_index(inplace=False))

    return df,dfsum          # if no ydf (target, for prediction) exists: thease output DataFrames are set to None

# *******************************
def all_nodata_remove(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame,pd.DataFrame]:

    """
        Removes nodata sample (by the way all inf values).
        nodat_remove must be used befor separation.
            Attention: QGIS Null for cells in a string-column will be not interpreted as None, 
            just string with length 0 (empty cell)
    Args:
        df (Pandas DataFrame)
    Returns:
        - pandas DataFrame: dataframe without nodata values
        - pandas DataFrame: nodatamask containing True for rows which are droped 
                            because of some nodata cells
    """

    df,dfsum =  _all_nodata_remove(                # np2: nodatamask
        df = df
    )
    return df,dfsum

