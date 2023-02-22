import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from typing import Optional, Tuple, Union, List, Literal

from eis_toolkit.exceptions import InvalidParameterValueException, InvalidInputDataException
from eis_toolkit.transformations import utils
from eis_toolkit.checks import parameter

# Core functions
def _winsorize_core(  # type: ignore[no-any-unimported]
    data_array: np.ndarray,
    limits: Tuple[int | float | None, int | float | None] = Tuple,
    replace_type: Literal['absolute', 'percentiles'] = Literal,
    replace_values: Optional[Tuple[int | float | None, int | float | None]] = None,
    replace_position: Optional[Literal['inside', 'outside']] = None,
    nodata_value: Optional[int | float] = None,
) -> Tuple[np.ndarray, Union[int, float, None], Union[int, float, None]]:  
    
    limit_min = limits[0]
    limit_max = limits[1]

    out_array = utils.replace_nan(data_array=data_array, nodata_value=nodata_value, set_nan=True)
    out_array = np.ma.array(out_array, mask=np.isnan(out_array))
    
    if replace_type == 'absolute':
        replace_lower = replace_values[0]
        replace_upper = replace_values[1]
        
        if limit_min is not None: out_array[out_array < limit_min] = replace_lower
        if limit_max is not None: out_array[out_array > limit_max] = replace_upper
    
    if replace_type == 'percentiles':
        if replace_position == 'inside':
            method_upper_interval = 'lower'
            method_lower_interval = 'higher'
        elif replace_position == 'outside':
            method_lower_interval = 'higher'
            method_upper_interval = 'lower'
            
        replace_lower = np.nanpercentile(out_array.data, limit_min, method=method_lower_interval) if limit_min is not None else None
        replace_upper = np.nanpercentile(out_array.data, 100-limit_max, method=method_upper_interval) if limit_max is not None else None
            
        if limit_min is not None: out_array[out_array < replace_lower] = replace_lower
        if limit_max is not None: out_array[out_array > replace_upper] = replace_upper

    out_array = out_array.data
    out_array = utils.replace_nan(data_array=out_array, nodata_value=nodata_value, set_value=True)
    
    return out_array, replace_lower, replace_upper


# Call functions
def _winsorize_raster(  # type: ignore[no-any-unimported]
    in_data: rasterio.io.DatasetReader,
    bands: Optional[List[int]] = None,
    limits: List[Tuple[int | float | None, int | float, None]] = List[Tuple],
    replace_type: Literal['absolute', 'percentiles'] = Literal,
    replace_values: Optional[List[Tuple[int | float | None, int | float, None]]] = None,
    replace_position: Optional[List[Literal['inside', 'outside']]] = None,
    nodata: Optional[List[int | float | None]] = None,
    method: Literal['replace', 'extract'] = 'replace',
) -> Tuple[np.ndarray, dict, dict]:
    raster = in_data
    
    if not bands: bands = list(range(1, raster.count + 1))
    
    if replace_type == 'absolute': 
        expanded_args = utils.expand_args(selection=bands, replace_values=replace_values)
        replace_values = expanded_args['replace_values']
    
    if replace_type == 'percentiles': 
        expanded_args = utils.expand_args(selection=bands, replace_position=replace_position)
        replace_position = expanded_args['replace_position']
        
    expanded_args = utils.expand_args(selection=bands, nodata=nodata, limits=limits, replace_type=replace_type)
    nodata = expanded_args['nodata']
    limits = expanded_args['limits']
    replace_type = expanded_args['replace_type']

    out_array, out_meta, out_meta_nodata, bands_idx = utils.read_raster(raster=raster, selection=bands, method=method)
    out_settings = {} 

    for i, band_idx in enumerate(bands_idx):
            nodata_value = out_meta_nodata[i] if not nodata or nodata[i] is None else nodata[i]
            
            replacement = replace_values[i] if replace_values else None
            replacement_position = replace_position[i] if replace_position else None
            
            out_array[band_idx], replace_lower, replace_upper = _winsorize_core(data_array=out_array[band_idx],
                                                                                limits=limits[i],
                                                                                replace_type=replace_type,
                                                                                replace_values=replacement,
                                                                                replace_position=replacement_position,
                                                                                nodata_value=nodata_value)
            
            current_transform = f'transform {band_idx + 1}'
            current_settings = {'band_origin': bands[i],
                                'limit_min': limits[i][0],
                                'limit_max': limits[i][1],
                                'replace_type': replace_type,
                                'replace_lower': round(replace_lower, ndigits=12) if replace_lower is not None else None,
                                'replace_upper': round(replace_upper, ndigits=12) if replace_upper is not None else None,
                                'replace_position': replace_position[i] if replace_position else None,
                                'nodata_meta': out_meta_nodata[i],
                                'nodata_used': nodata_value}
            out_settings[current_transform] = current_settings

    return out_array, out_meta, out_settings


def _winsorize_table(  # type: ignore[no-any-unimported]
    in_data: Union[pd.DataFrame, gpd.GeoDataFrame],
    columns: Optional[List[int]] = None,
    limits: List[Tuple[int | float | None, int | float, None]] = List[Tuple],
    replace_type: Literal['absolute', 'percentiles'] = Literal,
    replace_values: Optional[List[Tuple[int | float | None, int | float, None]]] = None,
    replace_position: Optional[List[Literal['inside', 'outside']]] = None,
    nodata: Optional[List[int | float | None]] = None,
) -> Tuple[np.ndarray, dict]:
    dataframe = in_data
    
    out_array, out_column_info, selection = utils.df_to_input_ordered_array(dataframe, columns)
    
    if replace_type == 'absolute': 
        expanded_args = utils.expand_args(selection=selection, replace_values=replace_values)
        replace_values = expanded_args['replace_values']
    
    if replace_type == 'percentiles': 
        expanded_args = utils.expand_args(selection=selection, replace_position=replace_position)
        replace_position = expanded_args['replace_position']
    
    expanded_args = utils.expand_args(selection=selection, nodata=nodata, limits=limits, replace_type=replace_type)
    nodata = expanded_args['nodata']
    limits = expanded_args['limits']
    replace_type = expanded_args['replace_type']
 
    out_settings = {}
    
    for i, column in enumerate(selection):
        nodata_value = nodata[i] if nodata is not None else None
        
        replacement = replace_values[i] if replace_values else None
        replacement_position = replace_position[i] if replace_position else None
        
        out_array[i], replace_lower, replace_upper = _winsorize_core(data_array=out_array[i],
                                                                     limits=limits[i],
                                                                     replace_type=replace_type,
                                                                     replace_values=replacement,
                                                                     replace_position=replacement_position,
                                                                     nodata_value=nodata_value)

        current_transform = f'transform {i + 1}'
        current_settings = {'original_column_name': column,
                            'original_column_index': dataframe.columns.get_loc(column),
                            'array_index': i,
                            'limit_min': limits[i][0],
                            'limit_max': limits[i][1],
                            'replace_type': replace_type,
                            'replace_lower': round(replace_lower, ndigits=12) if replace_lower is not None else None,
                            'replace_upper': round(replace_upper, ndigits=12) if replace_upper is not None else None,
                            'replace_position': replace_position[i] if replace_position else None,
                            'nodata_used': nodata_value}
        
        out_settings[current_transform] = current_settings

    return out_array, out_column_info, out_settings

# Call functions
def winsorize(  # type: ignore[no-any-unimported]
    in_data: Union[rasterio.io.DatasetReader, pd.DataFrame, gpd.GeoDataFrame],
    selection: Optional[List[int]] = None,
    limits: List[Tuple[int | float | None, int | float, None]] = List[Tuple],
    replace_type: Literal['absolute', 'percentiles'] = Literal,
    replace_values: Optional[List[Tuple[int | float | None, int | float, None]]] = None,
    replace_position: Optional[List[Literal['inside', 'outside']]] = None,
    nodata: Optional[List[int | float | None]] = None,
    method: Optional[Literal['replace', 'extract']] = None,
) -> Tuple[np.ndarray, dict, dict]:
    """Replace values below/above a given treshold or percentile value in a data set.
     
    Takes care of data with NoData values, input can be
    - None
    - user-defined
    If None, NoData will be read from raster metadata.
    If specified, user-input will be preferred.

    Works for multiband raster and multi-column dataframes.
    If no band/column selection specified, all bands/columns will be used.
    
    If only one NoData value, limit or replace tuple is specified, it will be used for all (selected) bands/columns.
    Contributed parameters will generally be applied for each band/column separately. This way, data can easily be transformed 
    by the same parameters or with different parameters for each band/column (values corresponding to each band/column).
    However, a mix of types ('absolute'/'percentile') is not allowed.
    
    If method is 'replace', selected bands/colums will be overwritten. Order of bands will not be changed in the output.
    If method is 'extract', only selected bands/columns will be returned. Order in the output corresponds to the order of the specified selection.
         
    For option 'absolute' values:
    -----------------------------
    Replaces values between minimum and lower treshold if limit_min is given.
    Replaces values between upper treshold and maximum if limit_max is given.
    Works both one-sided and two-sided but does not replace any values if no limits exist.
    
    Length of limits and replace_values is depended of the selected bands.
    If limits contains only one value, replace_values must be either 1 or length of the selection.
    If replace_values contains only one value, limits must be either 1 or length of the selection.
    
    Specific arguments: limits, replace_values
    
    For option 'percentiles':
    -------------------------
    Replaces values between minimum and lower percentile if limit_min is given.
    Replaces values between upper percentile and maximum if limit_max is given.
    Works both one-sided and two-sided but does not replace any values if no limits exist.
    The absolute treshold value will be re-calculated for every single band based on the user-defined percentiles.
    Percentile-values have to be in range [0, 100].
    
    Values are symmetric, meaning that a value of limit_min = 10 corresponds to the interval [min, 10%] 
    and limit_max = 10 corresponds to the intervall [90%, max]. Given this logic, limit_min = 0 refers 
    to the minimum and limit_max = 0 to the data maximum. 
    
    A replacement value corresponds to the data point which is nearest to the calculated 
    percentile value. Because the calculation of percentiles is ambiguous, the user can choose whether
    a replacement value should be taken from a data point located inside or outside the respective
    interval. If inside, data will be replaced with a value from within the computed interval. 
    If outside, data will be replaced with a value from outside the computed interval.
    
    Specific arguments: limits, replace_position

    Args:
        in_data (rasterio.io.DatasetReader, pd.DataFrame, gpd.GeoDataFrame): Data object to be transformed.
        selection (List[int | str], optional): Bands [int] or columns [str] to be processed. Defaults to None.
        limits (List[Tuple[int | float | None, int | float, None]]): Tuple for tresholds below/above values will be replaced (min, max). 
        replace_type (Literal['absolute', 'percentiles']): Option whether to replace by absolute or percentile values. Applied on whole data set.
        replace_values (List[Tuple[int | float | None, int | float, None]], optional): Tuple containing the new values (lower, upper). Req. only for replace_type = 'absolute'. Defaults to None.
        replace_position (List[Literal['inside', 'outside']], optional): List containing whether to use inner our outer values of an interval. Req. only for replace_type = 'percentiles'. Defaults to None.
        nodata (List[int | float], optional): NoData values to be considered. Defaults to None.
        method (Optional[Literal['replace', 'extract']]): Applied method for data output. For raster data only. Defaults to none.
        
    Returns:
        out_array (np.ndarray): The transformed data.
        out_meta (dict): Updated metadata with new band count. Only for raster data.
        out_column_info (dict): Dictionary containing transformable and non-transformable columns and geometry information. Only for pandas/geopandas data.
        out_settings (dict): Return of the input settings related to the new output.

    Raises:
        InvalidParameterValueException: The input contains invalid values.
    """       
    valids = parameter.check_selection(in_data, selection)
    valids.append(('Limits length', parameter.check_parameter_length(selection, limits, choice=1)))
    valids.append(('Limits values length', all(parameter.check_parameter_length(parameter=item, choice=2) for item in limits)))
    valids.append(('Limits values data type', all([all(isinstance(element, Union[int, float, None]) for element in item) for item in limits])))
    valids.append(('Limits values NoneType count', max([sum(element is None for element in item) for item in limits]) < 2))
    valids.append(('NoData length', parameter.check_parameter_length(selection, nodata, choice=1, nodata=True)))
    valids.append(('Replace type', replace_type == 'absolute' or replace_type == 'percentiles'))

    if nodata is not None: 
        valids.append(('NoData data type', all(isinstance(item, Union[int, float, None]) for item in nodata)))
    
    if replace_type == 'absolute':
        valids.append(('Limits values order', all(parameter.check_numeric_minmax_location(item) for item in limits if not None in item)))
            
        if replace_values is not None:
            valids.append(('Replace length', parameter.check_parameter_length(selection, replace_values, choice=1)))           
            valids.append(('Replace values data type', min([all(isinstance(element, Union[int, float, None]) for element in item) for item in replace_values])))
            valids.append(('Replace values length', all(parameter.check_parameter_length(parameter=item, choice=2) for item in replace_values)))
            valids.append(('Replace values NoneType count', max([sum(element is None for element in item) for item in replace_values]) < 2))
            valids.append(('Replace values NoneType position', parameter.check_none_positions(limits) == parameter.check_none_positions(replace_values)))
        elif replace_values is None:
            valids.append(('Replace values NoneType', False))
        
    if replace_type == 'percentiles':
        valids.append(('Limits value sum', all([sum(item) < 100 for item in limits if not None in item])))
        valids.append(('Limits value lower', all([0 < item[0] < 100 and item[0] != 0 for item in limits if item[0] is not None])))
        valids.append(('Limits value upper', all([0 < item[1] < 100 and item[1] != 0 for item in limits if item[1] is not None])))

        if replace_position is not None:
            valids.append(('Replace position length', parameter.check_parameter_length(selection, replace_position, choice=1)))
            valids.append(('Replace position data type', all(isinstance(item, str) for item in replace_position)))
            valids.append(('Replace position value', all(item == 'inside' or item == 'outside' for item in replace_position)))
        elif replace_position is None:
            valids.append(('Replace position NoneType', False))

    if isinstance(in_data, rasterio.DatasetReader):
        valids.append(('Output method', method == 'replace' or method == 'extract'))

        for item in valids:
            error_msg, validation = item
            
            if validation == False:
                raise InvalidParameterValueException(error_msg)
       
        out_array, out_meta, out_settings = _winsorize_raster(in_data=in_data,
                                                              bands=selection,
                                                              limits=limits,
                                                              replace_type=replace_type,
                                                              replace_values=replace_values,
                                                              replace_position=replace_position,
                                                              nodata=nodata,
                                                              method=method)
    
        return out_array, out_meta, out_settings
    
    if isinstance(in_data, Union[pd.DataFrame, gpd.GeoDataFrame]):
        for item in valids:
            error_msg, validation = item
            
            if validation == False:
                raise InvalidParameterValueException(error_msg)
       
        out_array, out_column_info, out_settings = _winsorize_table(in_data=in_data,
                                                                    columns=selection,
                                                                    limits=limits,
                                                                    replace_type=replace_type,
                                                                    replace_values=replace_values,
                                                                    replace_position=replace_position,
                                                                    nodata=nodata)
    
        return out_array, out_column_info, out_settings