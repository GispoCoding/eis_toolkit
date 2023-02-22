import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from typing import Optional, Tuple, Union, List, Literal

from eis_toolkit.exceptions import InvalidParameterValueException, InvalidInputDataException
from eis_toolkit.transformations import utils
from eis_toolkit.checks import parameter


# Core functions
def _sigmoid_transform_core(  # type: ignore[no-any-unimported]
    data_array: np.ndarray,
    new_range: Tuple[int | float, int | float] = (0, 1),
    shift: Tuple[int | float, int | float] = (0, 0),
    slope: Union[int, float] = 1,
    nodata_value: Optional[int | float] = None,
) -> np.ndarray:
    
    out_array = utils.replace_nan(data_array=data_array, nodata_value=nodata_value, set_nan=True)
    out_array[np.isinf(out_array)] = np.nan
    
    lower = new_range[0]
    upper = new_range[1]
    shift_x = shift[0]
    shift_y = shift[1]
    
    out_array = lower + (upper - lower) * (1 / (1 + np.exp(-slope*(out_array - shift_x)))) - shift_y
    
    out_array = utils.replace_nan(data_array=out_array, nodata_value=nodata_value, set_value=True)

    return out_array


# Call functions
def _sigmoid_transform_raster(  # type: ignore[no-any-unimported]
    in_data: rasterio.io.DatasetReader,
    bands: Optional[List[int]] = None,
    new_range: List[Tuple[int | float, int | float]] = [(0, 1)],
    shift: List[Tuple[int | float, int | float]] = [(0, 0)],
    slope: List[Union[int, float]] = [1],
    nodata: Optional[List[int | float | None]] = None,
    method: Literal['replace', 'extract'] = 'replace',
) -> Tuple[np.ndarray, dict, dict]:
        raster = in_data
        
        if not bands: bands = list(range(1, raster.count + 1))
        
        expanded_args = utils.expand_args(selection=bands, nodata=nodata, new_range=new_range, shift=shift, slope=slope)
        nodata = expanded_args['nodata']
        new_range = expanded_args['new_range']
        shift = expanded_args['shift']
        slope = expanded_args['slope']
    
        out_array, out_meta, out_meta_nodata, bands_idx = utils.read_raster(raster=raster, selection=bands, method=method)
        out_settings = {}

        for i, band_idx in enumerate(bands_idx):
            nodata_value = out_meta_nodata[i] if not nodata or nodata[i] is None else nodata[i]
            
            out_array[band_idx] = _sigmoid_transform_core(data_array=out_array[band_idx],
                                                          new_range=new_range[i],
                                                          shift=shift[i],
                                                          slope=slope[i],
                                                          nodata_value=nodata_value)
            
            current_transform = f'transform {band_idx + 1}'
            current_settings = {'band_origin': bands[i],
                                'scaled_min': new_range[i][0],
                                'scaled_max': new_range[i][1],
                                'shift_x': shift[i][0],
                                'shift_y': shift[i][1],
                                'slope': slope[i],
                                'nodata_meta': out_meta_nodata[i],
                                'nodata_used': nodata_value}
            
            out_settings[current_transform] = current_settings

        return out_array, out_meta, out_settings


def _sigmoid_transform_table(  # type: ignore[no-any-unimported]
    in_data: Union[pd.DataFrame, gpd.GeoDataFrame],
    columns: Optional[List[int]] = None,
    new_range: List[Tuple[int | float, int | float]] = [(0, 1)],
    shift: List[Tuple[int | float, int | float]] = [(0, 0)],
    slope: List[Union[int, float]] = [1],
    nodata: Optional[List[int | float | None]] = None,
) -> Tuple[np.ndarray, dict, dict]:
    dataframe = in_data
    
    out_array, out_column_info, selection = utils.df_to_input_ordered_array(dataframe, columns)
    
    expanded_args = utils.expand_args(selection=selection, nodata=nodata, new_range=new_range, shift=shift, slope=slope)
    nodata = expanded_args['nodata']
    new_range = expanded_args['new_range']
    shift = expanded_args['shift']
    slope = expanded_args['slope']
 
    out_settings = {}
    
    for i, column in enumerate(selection):
        nodata_value = nodata[i] if nodata is not None else None
        
        out_array[i] = _sigmoid_transform_core(data_array=out_array[i],
                                               new_range=new_range[i],
                                               shift=shift[i],
                                               slope=slope[i],
                                               nodata_value=nodata_value)

        current_transform = f'transform {i + 1}'
        current_settings = {'original_column_name': column,
                            'original_column_index': dataframe.columns.get_loc(column),
                            'array_index': i,
                            'scaled_min': new_range[i][0],
                            'scaled_max': new_range[i][1],
                            'shift_x': shift[i][0],
                            'shift_y': shift[i][1],
                            'slope': slope[i],
                            'nodata_used': nodata_value}
        
        out_settings[current_transform] = current_settings

    return out_array, out_column_info, out_settings
 
  
def sigmoid_transform(  # type: ignore[no-any-unimported]
    in_data: Union[rasterio.io.DatasetReader, pd.DataFrame, gpd.GeoDataFrame],
    selection: Optional[List[int]] = None,
    new_range: List[Tuple[int | float, int | float]] = [(0, 1)],
    shift: List[Tuple[int | float, int | float]] = [(0, 0)],
    slope: List[Union[int, float]] = [1],
    nodata: Optional[List[int | float | None]] = None,
    method: Optional[Literal['replace', 'extract']] = None,
) -> Tuple[np.ndarray, dict, dict]:
    """Z-score normalization.
    
    Transforms input data based on the sigmoid function.
        
    Takes care of data with NoData values, input can be
    - None
    - user-defined
    If None, NoData will be read from raster metadata.
    If specified, user-input will be preferred.
    
    If infinity values occur, they will be replaced by NaN.
    
    Works for multiband raster and multi-column dataframes.
    If no band/column selection specified, all bands/columns will be used.
    
    If only one NoData, new_range, shift and slope value is specified, it will be used for all (selected) bands.
    Contributed parameters will generally be applied for each band/column separately. This way, data can easily be transformed 
    by the same parameters or with different parameters for each band/column (values corresponding to each band/column).
    
    If method is 'replace', selected bands/colums will be overwritten. Order of bands will not be changed in the output.
    If method is 'extract', only selected bands/columns will be returned. Order in the output corresponds to the order of the specified selection.
    
    Args:
        in_data (rasterio.io.DatasetReader, pd.DataFrame, gpd.GeoDataFrame): Data object to be transformed.
        selection (List[int | str], optional): Bands [int] or columns [str] to be processed. Defaults to None.
        new_range: (List[Tuple[int | float, int | float]]): List containing the range tuple (min, max) for new minimum and maximum. Defaults to (0, 1).
        shift: (List[Tuple[int | float, int | float]]): List containing the shift (x, y) of the sigmoid function. Defaults to (0, 0).
        slope: (List[Union[int, float]]): List containing the adjustment value for the slope of the sigmoid functioin. Defaults to (1).
        nodata (List[Union[int, float], optional): NoData values to be considered. Defaults to None.
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
    valids.append(('New range length', parameter.check_parameter_length(selection, new_range, choice=1)))
    valids.append(('Shift length', parameter.check_parameter_length(selection, shift, choice=1)))
    valids.append(('Slope length', parameter.check_parameter_length(selection, slope, choice=1)))
    valids.append(('NoData length', parameter.check_parameter_length(selection, nodata, choice=1, nodata=True)))
    valids.append(('New range values data type', all([all(isinstance(element, Union[int, float]) for element in item) for item in new_range])))
    valids.append(('New range values length', all(parameter.check_parameter_length(parameter=item, choice=2) for item in new_range)))
    valids.append(('New range values order', all(parameter.check_numeric_minmax_location(item) for item in new_range)))
    valids.append(('Shift values data type', all([all(isinstance(element, Union[int, float]) for element in item) for item in shift])))
    valids.append(('Shift values length', all(parameter.check_parameter_length(parameter=item, choice=2) for item in shift)))
    valids.append(('Slope values data type', all(isinstance(item, Union[int, float]) for item in slope)))
    
    if nodata is not None: 
        valids.append(('NoData data type', all(isinstance(item, Union[int, float, None]) for item in nodata)))
    
    if isinstance(in_data, rasterio.DatasetReader):       
        valids.append(('Output method', method == 'replace' or method == 'extract'))

        for item in valids:
            error_msg, validation = item
            
            if validation == False:
                raise InvalidParameterValueException(error_msg)

        out_array, out_meta, out_settings = _sigmoid_transform_raster(in_data=in_data,
                                                                      bands=selection,
                                                                      new_range=new_range,
                                                                      shift=shift,
                                                                      slope=slope,
                                                                      nodata=nodata,
                                                                      method=method)
        
        return out_array, out_meta, out_settings
    
    if isinstance(in_data, Union[pd.DataFrame, gpd.GeoDataFrame]):
        for item in valids:
            error_msg, validation = item
            
            if validation == False:
                raise InvalidParameterValueException(error_msg)
        
        out_array, out_column_info, out_settings = _sigmoid_transform_table(in_data=in_data,
                                                                            columns=selection,
                                                                            new_range=new_range,
                                                                            shift=shift,
                                                                            slope=slope,
                                                                            nodata=nodata)
        
        return out_array, out_column_info, out_settings
    
