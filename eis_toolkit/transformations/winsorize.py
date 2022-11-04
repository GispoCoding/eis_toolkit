import numpy as np
import pandas as pd
import rasterio
from typing import Optional, Tuple, Union

from eis_toolkit.checks.parameter import check_numeric_value_sign
from eis_toolkit.exceptions import NumericValueSignException


# The core functions for winsorizing.
def _winsorize_fixed_values(
  data_array: np.ndarray,
  limit_min: Optional[float] = None,
  limit_max: Optional[float] = None,
  replace_value_min: Optional[float] = None,
  replace_value_max: Optional[float] = None,
  nan_value: Optional[Union[str, float]] = None,
) -> np.ndarray:
    
    out_array = np.where(data_array == nan_value, np.nan, data_array) if nan_value else data_array
    out_array = np.where(out_array < limit_min, replace_value_min, out_array) if limit_min and replace_value_min else out_array
    out_array = np.where(out_array > limit_max, replace_value_max, out_array) if limit_max and replace_value_max else out_array
    out_array = np.where(out_array == np.nan, nan_value, out_array) if nan_value else out_array
       
    return out_array


def _winsorize_percentile_values(
    data_array: np.ndarray,
    limit_min: Optional[float] = None,
    limit_max: Optional[float] = None,
    replace_value_location: bool = True, 
    nan_value: Optional[Union[str, float]] = None,
) -> Tuple[np.ndarray, dict]:
         
    out_array = np.where(data_array == nan_value, np.nan, data_array) if nan_value else data_array
    
    method_lower_interval = "upper" if replace_value_location == True else "lower"
    method_upper_interval = "lower" if replace_value_location == True else "upper"
    
    lower_treshold = np.nanpercentile(out_array, limit_min, method = method_lower_interval) if limit_min else None
    upper_treshold = np.nanpercentile(out_array, 100-limit_max, method = method_upper_interval) if limit_max else None
    
    out_array = np.where(out_array < lower_treshold, lower_treshold, out_array) if limit_min else out_array
    out_array = np.where(out_array > upper_treshold, upper_treshold, out_array) if limit_max else out_array 
    out_array = np.where(out_array == np.nan, nan_value, out_array) if nan_value else out_array   
    
    out_dict = {
        "replacement_lower": lower_treshold,
        "replacement_upper": upper_treshold,
    }
    
    return out_array, out_dict


# The use-case functions.
def winsorize_raster_fixed(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    limit_min: Optional[float] = None,
    limit_max: Optional[float] = None,
    replace_value_min: Optional[float] = None,
    replace_value_max: Optional[float] = None,
    nan_value: Optional[Union[str, float]] = None,
    nan_value_ignore: bool = False,
):
    """Replace values below/above a given treshold in a raster data set.
    
    Replaces values between minimum and lower treshold if limit_min is given.
    Replaces values between upper treshold and maximum if limit_max is given.
    Works both one-sided and two-sided but does not replace any values if no limits exist.
    
    Takes care of raster data with NoData values, which can, be user-defined, read-in from raster 
    meta data if available (default) or ignored.If given, user input will be prefered over the 
    raster metadata.
    

    Args:
        raster (rasterio.io.DatasetReader): Raster to be transformed. No multiband support yet!
        limit_min (float): Treshold below all values will be replaced. Defaults to None.
        limit_max (float): Treshold above all values will be replaced. Defaults to None.
        replace_value_min (float): Repeacement value for lower interval. Defaults to None.
        replace_value_max (float): Replacement value for upper interval. Defaults to None.
        nan_value (str, float): NoData value of a raster data set. Defaults to None.
        nan_value_ignore (bool): Switch to ignore both input and raster NoData values. Defaults to false. 

    
    Returns:
        out_image (np.ndarray): The transformed raster data.
        out_meta (dict): The raster metadata.

    Raises:
        InvalidParameterValueException: The input contains invalid values.
    """
    
    out_meta = raster.meta.copy()
    
    if nan_value_ignore == False:
        nan_value = out_meta["nodata"] if nan_value is not None else nan_value
    
    out_image = _winsorize_fixed_values(
        data_array=raster.read(),
        limit_min=limit_min,
        limit_max=limit_max,
        replace_value_min=replace_value_min,
        replace_value_max=replace_value_max,
        nan_value=nan_value,
    )
    return  out_image, out_meta

def winsorize_raster_percentiles(  # type: ignore[no-any-unimported]
    
):
    """Replace values below/above a given treshold in a raster data set.
    
    Replaces values between minimum and lower percentile if limit_min is given.
    Replaces values between upper percentile and maximum if limit_max is given.
    Works both one-sided and two-sided but does not replace any values if no limits exist.
    Percentiles have to be in range [0, 100].
    
    Treshold values are symmetric, so that a value of limit_min = 10 corresponds to the interval 
    [min, 10%] and limit_max = 10 corresponds to the intervall [90%, max] of the data set.
    limit_min = 0 refers to the minimum and limit_max = 0 to the data maximum. 
    
    A replacement value corresponds to the data point which is nearest to the calculated 
    percentile value. Because the calculation of percentiles is ambiguous, user can choose whether
    a replacement value should be taken from a data point inside or outside the respective interval. 
    For inside, data will be replaced with a value from within the computed interval. For outside, 
    data will be replaced with a value from outside the computed interval. This is the default.
    
    Takes care of raster data with NoData values, which can, be user-defined, read-in from raster 
    meta data if available (default) or ignored.If given, user input will be prefered over the 
    raster metadata.
    

    Args:
        raster (rasterio.io.DatasetReader): Raster to be transformed. No multiband support yet!
        limit_min (float): Treshold below all values will be replaced. Defaults to None.
        limit_max (float): Treshold above all values will be replaced. Defaults to None.
        replacement_value_location (bool): Use replacement from inside or outside. Defaults to True (outside).
        nan_value (str, float): NoData value of a raster data set. Defaults to None.
        nan_value_ignore (bool): Switch to ignore both input and raster NoData values. Defaults to false. 


    Returns:
        out_image (np.ndarray): The transformed raster data.
        out_meta (dict): The raster metadata.
        out_replacement (dict): The computed replacement values.

    Raises:
        InvalidParameterValueException: The input contains invalid values.
    """

    return ...



# - **Requirements**
#     - core functions just for winsorizing execution
#     - checks before executing one of the core functions
#         - are there two open intervals ?
#         - does the number of replacement-values correspond to the intervals ? (only fixed values; percentiles will have default values)
#         - is min > max ?
#         - 0 and 100 will note make any difference for percentiles (no change) so no check needed
#         - is interval okay? 0 - 100 (since using percentiles, 0-1 would be for quantiles) -> intervals are not allowed to intersect each other
        
#     - choice wether to prepare (and save) raster data or data frames also prior to core function execution
#     - need possibility for open intervals [x, None], [None, x]
#     - Replace both percentile or fixed values
#     - If percentile, replacement-defaults are the corresponding percentile values, but fixed replacements can be user-defined if not None (optional)
#     - If fixed values, replacements are not optional and need to be user-defined inputs
#     - Option for using value within or outside the choosen interval
#     - does not handle multiband-raster
#     - but might handle multiple inputs (list of file paths) for raster
#     - might also work for multiple columns in a data frame
#     - keep or delete old columns in a df ? users choice ?