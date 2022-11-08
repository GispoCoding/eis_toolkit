import numpy as np
import pandas
import geopandas
import rasterio
from typing import Optional, Tuple, Union, List

from eis_toolkit.checks.parameter import check_numeric_value_sign
from eis_toolkit.exceptions import NumericValueSignException, InvalidParameterValueException


# The core functions for winsorizing.
def _winsorize_fixed_values(
  data_array: np.ndarray,
  limit_min: Optional[float] = None,
  limit_max: Optional[float] = None,
  replace_lower: Optional[float] = None,
  replace_upper: Optional[float] = None,
  nan_value: Optional[Union[str, int, float]] = None,
) -> np.ndarray:
    
    out_array = np.where(data_array == nan_value, np.nan, data_array) if nan_value else data_array
    out_array = np.where(out_array < limit_min, replace_lower, out_array) if limit_min and replace_lower else out_array
    out_array = np.where(out_array > limit_max, replace_upper, out_array) if limit_max and replace_upper else out_array
    out_array = np.where(np.isnan(out_array), nan_value, out_array) if nan_value else out_array
       
    return out_array


def _winsorize_percentile_values(
    data_array: np.ndarray,
    limit_min: Optional[float] = None,
    limit_max: Optional[float] = None,
    replace_value_location: bool = True,
    nan_value: Optional[Union[str, int, float]] = None,
) -> Tuple[np.ndarray, dict]:
    
    out_array = np.where(data_array == nan_value, np.nan, data_array) if nan_value else data_array
    
    method_lower_interval = "upper" if replace_value_location == True else "lower"
    method_upper_interval = "lower" if replace_value_location == True else "upper"
    
    lower_treshold = np.nanpercentile(out_array, limit_min, method = method_lower_interval) if limit_min else None
    upper_treshold = np.nanpercentile(out_array, 100-limit_max, method = method_upper_interval) if limit_max else None
    
    out_array = np.where(out_array < lower_treshold, lower_treshold, out_array) if limit_min else out_array
    out_array = np.where(out_array > upper_treshold, upper_treshold, out_array) if limit_max else out_array
    out_array = np.where(np.isnan(out_array), nan_value, out_array) if nan_value else out_array
    
    out_dict = {
        "replacement_lower": lower_treshold,
        "replacement_upper": upper_treshold,
    }
    
    return out_array, out_dict


# The use-case functions.
def winsorize_fixed_values(  # type: ignore[no-any-unimported]
    in_data: Union[rasterio.io.DatasetReader, pandas.DataFrame, geopandas.GeoDataFrame],
    bands: Optional[List[int]] = None,
    limit_min: Optional[float] = None,
    limit_max: Optional[float] = None,
    replace_lower: Optional[float] = None,
    replace_upper: Optional[float] = None,
    nan_value: Optional[Union[int, float]] = None,
    nan_value_ignore: bool = False,
) -> Tuple[np.ndarray, dict]:
    """Replace values below/above a given treshold in a raster data set.
    
    Replaces values between minimum and lower treshold if limit_min is given.
    Replaces values between upper treshold and maximum if limit_max is given.
    Works both one-sided and two-sided but does not replace any values if no 
    limits exist.
    
    Takes care of raster data with NoData values, which can be user-defined, 
    read-in from raster meta data (default) or ignored. If given, user input 
    will be prefered over the raster metadata.
    
    If bands are not given, all bands will be used for transformation. This 
    is the default. Contributed parameters will generally be applied for 
    each band.
    
    
    Args:
        in_data (rasterio.io.DatasetReader, pandas.DataFrame, geopandas.GeoDataFrame): Single data set to be transformed.
        bands (List[int], optional): Band numbers of raster data set to be processed. Defaults to None.
        limit_min (float, optional): Treshold below all values will be replaced. Defaults to None.
        limit_max (float, optional): Treshold above all values will be replaced. Defaults to None.
        replace_lower (float, optional): Replacement value for lower interval. Defaults to None.
        replace_upper (float, optional): Replacement value for upper interval. Defaults to None.
        nan_value (int, float, optional): NoData value of a raster data set. Defaults to None.
        nan_value_ignore (bool): Switch to ignore both input and raster NoData values. Defaults to false. 

    Returns:
        out_image (np.ndarray): The transformed raster data.
        out_meta (dict): The raster metadata.

    Raises:
        InvalidParameterValueException: The input contains invalid values.
    """
    
    if limit_min is None and limit_max is None:
        raise InvalidParameterValueException
    elif limit_min is not None and limit_max is not None and limit_min > limit_max:
        raise InvalidParameterValueException

    if isinstance(in_data, rasterio.io.DatasetReader):
        if bands is not None:
            if not isinstance(bands, list):
                raise InvalidParameterValueException
            elif not all(isinstance(band, int) for band in bands):
                raise InvalidParameterValueException
            if max(bands) > in_data.count:
                raise InvalidParameterValueException
            
        data_array = in_data.read() if bands is None else in_data.read(bands)
        out_meta = in_data.meta
    elif isinstance(in_data, pandas.DataFrame | geopandas.GeoDataFrame):
        print("Functionality to be added right here")    
        
    if nan_value_ignore == False:
        nan_value = nan_value if nan_value is not None else in_data.nodata
    else:
        nan_value = None

    out_array = _winsorize_fixed_values(
        data_array=data_array,
        limit_min=limit_min,
        limit_max=limit_max,
        replace_lower=replace_lower,
        replace_upper=replace_upper,
        nan_value=nan_value,
    )
    return out_array, out_meta

def winsorize_percentiles_raster(  # type: ignore[no-any-unimported]
    
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
    a replacement value should be taken from a data point located inside or outside the respective
    interval. If inside, data will be replaced with a value from within the computed interval. 
    If outside, data will be replaced with a value from outside the computed interval. This is 
    the default.
    
    Takes care of raster data with NoData values, which can be user-defined, read-in from raster 
    meta data if available (default) or ignored. If given, user input will be prefered over the 
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
    
    # Perzentilberechnung anpassen für multiband, sonst mglw. falsch
    # if check_numeric_value_sign(limit_min):
    #     raise NumericValueSignException
    # if check_numeric_value_sign(limit_max):
    #     raise NumericValueSignException
    # if limit_min > limit_max:
    #     raise InvalidNumericValueException

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