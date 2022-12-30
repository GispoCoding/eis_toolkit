"""
remove rows  delete samples with nodata cells 
State an Dezember 04 2022
@author: torchala 
""" 

### Stand: fast fertig, zu verbessern und zu prüfen (tests)
#    - durchtesten: float64, wie reagiert import_grid (OnHotEncoder) auf float-Eingaben?
#    - if multitarget will be possible for y-DataFrame (target), the Module has to be enhanced and checked
#    - Weitere Tests für andere Anwendunegn (nicht Raster, sondern shape, csv, ESRI-feature-classes..)
##   . Idee: ggf. von Anfang an, die target-Trainingsdaten mit in X durchziehen, da brauchen wir hier keine Sonderbehandlung

from typing import Tuple, Optional
import copy
import numpy as np 
import pandas as pd
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _nodata_remove(
    Xdf: pd.DataFrame
) -> Tuple[pd.DataFrame,pd.DataFrame]:     # das 2. df ist die neue nodata-mask (nur eine Spalte, ebenso viele reihen, wie die erste df)

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
        ydf  (Pandas DataFrame): is exists (for prediction)

    Returns:
        - pd.DataFrame: dataframe without nodata values
        - target DataFrame: for training purposes (in case of prediction target df will be Non)
        - nodatamask: DataFrame (bool, one column) containing True for rows which are droped because of some nodata cells

    """

    Xdf,dfsum =  _nodata_remove(                # np2: nodatamask
        Xdf = Xdf
    )
    return Xdf,dfsum

