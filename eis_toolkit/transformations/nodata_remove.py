
from typing import Tuple
import copy
import numpy as np 
import pandas as pd
from separation import *
from unification import *
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _nodata_remove(
    Xdf: pd.DataFrame
) -> Tuple[pd.DataFrame,pd.DataFrame]:     # das 2. df ist die neue nodata-mask (nur eine Spalte, ebenso viele reihen, wie die erste df)

    if len(Xdf.columns) == 0:
        raise InvalidParameterValueException ('***  DataFrame has no column')
    if len(Xdf.index) == 0:
        raise InvalidParameterValueException ('***  DataFrame has no rows')
        
    # datatype to float32 (on this way float64 come to np.inf)
    Xdf.loc[:,Xdf.dtypes=='float64'] = Xdf.loc[:,Xdf.dtypes=='float64'].astype('float32')
    Xdf.loc[:,Xdf.dtypes=='int64'] = Xdf.loc[:,Xdf.dtypes=='int64'].astype('int32')
    Xdf.replace([np.inf, -np.inf],np.nan,inplace=True)
    # for col in df.columns:
    #     if not is_numeric_dtype(df[col]):   # df[col].dtype != np.number:    # # empty strings cells will get numpy.nan
    #Xdf.replace(r'^\s*$',np.nan,regex=True,inplace=True)   # or: df.replace(r'\s+', np.nan, regex=True)

    # Xvdf, Xcdf, ydf = separation(Xdf = Xdf, fields = columns)
    # Xdfv = Xdfv.astype('float32')
    # Xdf.replace([np.inf, -np.inf], np.nan, inplace=True)
    #unification

    dfmask = Xdf.isna()      # dfmsk will be needed in module "export_grid" after prediction to add nodata-values in y after prediction
    dfsum = pd.DataFrame(dfmask.sum(axis=1))
    dfsum[dfsum>0] = True
    dfsum[dfsum==0] = False
    #dfsum.replace({1:True,0:False},inplace = True)
    Xdf.dropna(inplace = True)    # drops rows which contain missing values
    Xdf = copy.deepcopy(Xdf.reset_index(inplace=False))

    return Xdf,dfsum          # if no ydf (target, for prediction) exists: thease output DataFrames are set to None

# *******************************
def nodata_remove(
    Xdf: pd.DataFrame
) -> Tuple[pd.DataFrame,pd.DataFrame]:

    """
        Removes nodata sample (by the way all inf values).
        nodat_remove must be used befor separation.
            Attention: QGIS Null for cells in a string-column will be not interpreted as None, 
            just string with length 0 (empty cell)
    Args:
        Xdf (Pandas DataFrame)
    Returns:
        - pandas DataFrame: dataframe without nodata values
        - pandas DataFrame: nodatamask containing True for rows which are droped 
                            because of some nodata cells
    """

    Xdf,dfsum =  _nodata_remove(                # np2: nodatamask
        Xdf = Xdf
    )
    return Xdf,dfsum

