import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from typing import Optional, Tuple, Union, List, Literal

from eis_toolkit.exceptions import InvalidParameterValueException, InvalidInputDataException
from eis_toolkit.transformations import utils

# Core functions
def _winsorize_absolute_values(  # type: ignore[no-any-unimported]
    data_array: np.ndarray,
    limits: Tuple[Optional[int | float]] = Tuple,
    replacements: Tuple[Optional[int | float]]= Tuple,
    nodata_value: Union[int, float] = int,
) -> np.ndarray:
  
    limit_min = limits[0]
    limit_max = limits[1]
    replace_min = replacements[0]
    replace_max = replacements[1]
    
    out_array = np.where(data_array == nodata_value, np.nan, data_array) if not np.isnan(nodata_value) else data_array
    out_array = np.where(out_array < limit_min, replace_min, out_array) if limit_min else out_array
    out_array = np.where(out_array > limit_max, replace_max, out_array) if limit_max else out_array
    
    if not np.isnan(nodata_value): out_array = np.where(np.isnan(out_array), nodata_value, out_array)
    
    return out_array


def _winsorize_percentile_values(  # type: ignore[no-any-unimported]
    data_array: np.ndarray,
    limits: Tuple[Optional[int | float]] = Tuple,
    replace_position: Literal["inside", "outside"] = str,
    nodata_value: Union[int, float] = int,
) -> np.ndarray:
    
    if replace_position == "inside":
        method_upper_interval = "lower"
        method_lower_interval = "higher"
    elif replace_position == "outside":
        method_lower_interval = "higher"
        method_upper_interval = "lower"
         
    out_array = np.where(data_array == nodata_value, np.nan, data_array) if not np.isnan(nodata_value) else data_array
    
    limit_min = limits[0]
    limit_max = limits[1]
    
    if limit_min: lower_treshold = np.nanpercentile(out_array, limit_min, method = method_lower_interval)
    if limit_max: upper_treshold = np.nanpercentile(out_array, 100-limit_max, method = method_upper_interval)
    
    out_array = np.where(out_array < lower_treshold, lower_treshold, out_array) if limit_min else out_array
    out_array = np.where(out_array > upper_treshold, upper_treshold, out_array) if limit_max else out_array
    
    if not np.isnan(nodata_value): out_array = np.where(np.isnan(out_array), nodata_value, out_array)

    return out_array  

# Call functions
def _winsorize_raster(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    selection: Optional[List[int | float]] = None,
    limits_min: Optional[List[int | float | None]] = None,
    limits_max: Optional[List[int | float | None]] = None,
    replacements_type: Optional[List[Literal["absolute", "percentiles"]]] = None,
    replacements_lower: Optional[List[int | float | None]] = None,
    replacements_upper: Optional[List[int | float | None]] = None,
    replacements_position: Optional[List[Literal["inside", "outside"]]] = None,
    nodata_values: List[Optional[int | float]] = List,
) -> np.ndarray:

    out_array = raster.read()
    out_meta = raster.meta
    nodata_values = raster.nodatavals
    
    if not band_selection:
        band_selection = list(range(0, raster.count))
    
    for i in range(0, raster.count):
        nodata_value = raster.nodatavals[i] if not nodata_values[i] else nodata_values[i]     
        
        # if not replace_lower: replace_lower = limit_min
        # if not replace_upper: replace_upper = limit_max   
            
        # if not nodata_value: nodata_value = np.nan
        # if nodata_replace: nodata_value = np.nan
    
        if i + 1 == band_selection[i]:
            if replacements_type[i] == "absolute":
                out_array[i] = _winsorize_absolute_values(data_array=out_array[i],
                                                          limit_min=limits_min[i],
                                                          limit_max=limits_max[i],
                                                          replace_lower=replacements_lower[i],
                                                          replace_upper=replacements_upper[i],
                                                          nodata_value=nodata_value)


    return out_array

