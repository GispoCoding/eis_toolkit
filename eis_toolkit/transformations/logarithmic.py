import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from typing import Optional, Tuple, Union, List, Literal

from eis_toolkit.exceptions import InvalidParameterValueException, InvalidInputDataException
from eis_toolkit.transformations import utils

# Core functions
def _log_transform(  # type: ignore[no-any-unimported]
    data_array: np.ndarray,
    base: int = 2,
    nodata_value: Optional[int | float] = None,
) -> np.ndarray:
    
    out_array = utils._replace_nan(data_array=data_array, nodata_value=nodata_value, set_nan=True)
    out_array[np.isinf(out_array)] = np.nan
    
    if base == 2: out_array = np.log2(out_array)
    if base == 10: out_array = np.log10(out_array)

    out_array = utils._replace_nan(data_array=out_array, nodata_value=nodata_value, set_value=True)

    return out_array
