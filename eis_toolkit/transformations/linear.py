import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from typing import Optional, Tuple, Union, List, Literal

from eis_toolkit.exceptions import InvalidParameterValueException, InvalidInputDataException
from eis_toolkit.transformations import utils

# Core functions
def _z_score_normalization(  # type: ignore[no-any-unimported]
    data_array: np.ndarray,
    with_mean: bool = True,
    with_std: bool = True,
    nodata_value: Optional[int | float] = None,
) -> np.ndarray:
    
    out_array = utils.replace_nan(data_array=data_array, nodata_value=nodata_value, set_nan=True)
    out_array[np.isinf(out_array)] = np.nan
    
    mean = 0 if not with_mean else np.nanmean(out_array)
    std = 1 if not with_std else np.nanstd(out_array)
    out_array = (out_array - mean) / std

    out_array = utils.replace_nan(data_array=out_array, nodata_value=nodata_value, set_value=True)

    return out_array


def _min_max_scaling(  # type: ignore[no-any-unimported]
    data_array: np.ndarray,
    range: Tuple[int | float] = (0, 1),
    nodata_value: Optional[int | float] = None,
) -> np.ndarray:
    
    out_array = utils.replace_nan(data_array=data_array, nodata_value=nodata_value, set_nan=True)
    out_array[np.isinf(out_array)] = np.nan  
    
    min = np.nanmin(out_array)
    max = np.nanmax(out_array)
    scaled_min = range[0]
    scaled_max = range[1]
    
    scaler = (out_array - min) / (max - min)
    out_array = (scaler * (scaled_max - scaled_min)) + scaled_min
        
    out_array = utils.replace_nan(data_array=out_array, nodata_value=nodata_value, set_value=True)

    return out_array