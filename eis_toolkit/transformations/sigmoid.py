import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from typing import Optional, Tuple, Union, List, Literal

from eis_toolkit.exceptions import InvalidParameterValueException, InvalidInputDataException
from eis_toolkit.transformations import utils

# Core functions
def _sigmoid_transform(  # type: ignore[no-any-unimported]
    data_array: np.ndarray,
    range: Tuple[int | float] = (0, 1),
    shift: Tuple[int | float] = (0, 0),
    slope: Union[int, float] = 1,
    nodata_value: Optional[int | float] = None,
) -> np.ndarray:
    
    out_array = utils._replace_nan(data_array=data_array, nodata_value=nodata_value, set_nan=True)
    out_array[np.isinf(out_array)] = np.nan
    
    lower = range[0]
    upper = range[1]
    shift_x = shift[0]
    shift_y = shift[1]
    
    out_array = lower + (upper - lower) * (1 / (1 + np.exp(-slope*(out_array-shift_x)))) - shift_y

    out_array = utils._replace_nan(data_array=out_array, nodata_value=nodata_value, set_value=True)

    return out_array