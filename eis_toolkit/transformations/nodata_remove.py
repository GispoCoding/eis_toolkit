import copy

import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Tuple

from eis_toolkit.exceptions import InvalidParameterValueException


# *******************************
@beartype
def _nodata_remove(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # datatype to float32 (on this way some float64 values may come to np.inf)
    df.loc[:, df.dtypes == "float64"] = df.loc[:, df.dtypes == "float64"].astype("float32")
    df.loc[:, df.dtypes == "int64"] = df.loc[:, df.dtypes == "int64"].astype("int32")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    dfmask = (
        df.isna()
    )  # dfmsk will be needed in module "export_grid" after prediction to add nodata-values in y after prediction
    dfsum = pd.DataFrame(dfmask.sum(axis=1))
    dfsum[dfsum > 0] = True
    dfsum[dfsum == 0] = False
    df.dropna(inplace=True)  # drops rows which contain missing values
    df = copy.deepcopy(df.reset_index(inplace=False))

    return df, dfsum  # if no ydf (target, for prediction) exists: thease output DataFrames are set to None


# *******************************
@beartype
def nodata_remove(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
        Remove nodata sample (and by the way all inf values).

        nodata_remove.py must be used befor separation.py.
            Attention: QGIS Null for cells in a string-column will be not interpreted as None,
            just string with length 0 (empty cell)

    Args:
        Pandas DataFrame

    Returns:
        - pandas dataframe without nodata values (raws with nodata cells are deleted)
        - pandas dataframe as nodatamask containing
            - True for rows which are droped because of some nodata cells
            - False else
    """

    # Argument evaluation
    if len(df.columns) == 0:
        raise InvalidParameterValueException("DataFrame has no column")
    if len(df.index) == 0:
        raise InvalidParameterValueException("DataFrame has no rows")

    return _nodata_remove(df=df)
