"""
export prediction result: add to XDF
Created an Dezember 02 2022
@author: bernd torchala 
""" 

# zu conversions

#### fast fertig
    ###  Verbesserung/Erweiterung:
    ## - Bisher nur eine Spalte. 
    ##    Wenn df mehrere Spalten (multitarget) hat, entsteht ein multiband oder mehrere tiff?
## offene Fragen:     
##                und dann außerhalb zu Gtiff-Datei  ausgeben??
##                metadata wieder mit ausgeben?
##                multitarget
##                noch weitere Prüfungen (z. B. ohne nodatamask)
# ggf. wird für die Ergebnisspalte noch ein Name gebraucht. 

from typing import Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _export_featureclass(
    Xdfg: gpd.GeoDataFrame | pd.DataFrame,  # ydf has to add to XDF
    ydf: pd.DataFrame, 
    nanmask: Optional[pd.DataFrame] = None 
) -> pd.DataFrame:

    out = Xdfg
    # if nodataremovement was made:
    if nanmask is None:   # 
        out['result'] = ydf.to_numpy()
    else:
        # assemple a list out of the input dataframe ydf (one column) and the nodatamask-True-values: NaN
        v = 0
        lst = []
        for cel in nanmask.iloc[:,0]:
            if cel == True:
                lst.append(np.NaN)
            else:
                lst.append(ydf.iloc[v,0])     # .values.tolist())
                v += 1
        out['result'] = lst
        # append as result-column
    
    return out

# *******************************
def export_featureclass(
    Xdfg: gpd.GeoDataFrame | pd.DataFrame,  # ydf has to add to XDF
    ydf: pd.DataFrame, 
    nanmask: Optional[pd.DataFrame] = None 

) -> pd.DataFrame:

    """ Add the result column to an existing geopandas dataframe
    In case a nanmask is availabel (nan-cells for prediction input caused droped rows): 
    "True"-cells in nanmask lead to nodata-cells in the output dataframe (y)

    Args:
        Xdf (pandas DataFrame or GeoDataFrame): is primary feature table - input for prediction process,
        ydf (pandas DataFrame): is result of prediction,
        nodata (pandas DataFrame): marks rows witch are droped because of nodata in the prediction input.

    Returns:
        gpd.GeoDataFrame or pd.DataFrame 
    """

    out = _export_featureclass(
        Xdfg = Xdfg,
        ydf = ydf,
        nanmask  = nanmask
    )

    return out
