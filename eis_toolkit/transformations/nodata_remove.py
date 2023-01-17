
from typing import Tuple, Optional
import copy
import numpy as np 
import pandas as pd
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _nodata_remove(
    Xdf: pd.DataFrame
) -> Tuple[pd.DataFrame,pd.DataFrame]:     # 2. df is nodata-mask 

    # datatype to float32
    Xdf = Xdf.astype('float32')
    Xdf.replace([np.inf, -np.inf], np.nan, inplace=True)

    dfmask = Xdf.isna()      # dfmsk will be needed in module "export_grid" after prediction to add nodata-values in y after prediction
    dfsum = pd.DataFrame(dfmask.sum(axis=1))
    dfsum.replace({1:True,0:False},inplace = True)
    Xdf.dropna(inplace = True)    # drops rows which contain missing values
    Xdf = copy.deepcopy(Xdf.reset_index(inplace=False))

    return Xdf,dfsum          # if no ydf (target, for prediction) exists: thease output DataFrames are set to None

# *******************************
def nodata_remove(
    Xdf: pd.DataFrame
) -> Tuple[pd.DataFrame,pd.DataFrame]:

    """removes nodata sample (by the way all inf values)

    Args:
        Xdf (Pandas DataFrame)
        ydf (Pandas DataFrame): is exists (for prediction)

    Returns:
        - pandas DataFrame: dataframe without nodata values
        - pandas DataFrame: nodatamask containing True for rows which are droped 
                            because of some nodata cells

    """

    Xdf,dfsum =  _nodata_remove(                # np2: nodatamask
        Xdf = Xdf
    )
    return Xdf,dfsum

