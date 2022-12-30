
from typing import Optional
import numpy as np
import pandas as pd
# from sklearn.preprocessing import OneHotEncoder
# from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
def _export_grid(
    metadata: dict,
    df: pd.DataFrame,
    nanmask: Optional[pd.DataFrame] = None
    # wenn gtiff raus, dann wird noch ein Name gebraucht und eine Funktion pandas_to_raster
) -> np.ndarray:

    if nanmask is None:   # reshape with metadata width and hiegt (nonaodata-samples ar removed)
        # width = metadata.width
        # height = metadata.height
        ####??? noch prüfen, ob shape(df) = width * height ist
        out = df.to_numpy().reshape(metadata['height'],metadata['width'])
    else:
        # assemple a list out of the input dataframe ydf (one column) and the nodatamask-True-values: NaN
        v = 0
        lst = []
        for cel in nanmask.iloc[:,0]:
            if cel == True:
                lst.append(np.NaN)
            else:
                lst.append(df.iloc[v,0])     # .values.tolist())
                v += 1

        out = np.array(lst).reshape(metadata[0]['height'],metadata[0]['width'])

    return out

# *******************************
def export_grid(
    metadata: dict,
    df: pd.DataFrame,
    nanmask: Optional[pd.DataFrame] = None
) -> np.ndarray:

    """ reshape one column of a pandas df to a new pd dataframe with width and height. 
    In case a nanmask is availabel (nan-cells for prediction input caused droped rows): 
    "True"-cells in nanmask lead to nodata-cells in the output dataframe (y)
   metadata contains width and height values out of input grids for prediction.
   nodata marks rows witch are droped because of nodata in the prediction input.

    Args:
        metadata (dictionary): contains with and height values
        df (pandas DataFrame): is the input df comming from prediction-method 
        nanmask )pandas DataFrame): in case nodata-samples are removed during "nodata-replacement"

    Returns:
        pd.DataFrame: Raster converted to new columns of pandas dataframe
    """

    out = _export_grid(
    metadata = metadata,
    df = df,
    nanmask  = nanmask
    )

    return out

