import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from typing import Optional, Tuple, Union, List, Literal

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.transformations import utils
from eis_toolkit.checks import parameter


# Core functions
def _binarize_core(  # type: ignore[no-any-unimported]
    data_array: np.ndarray,
    threshold: Union[int, float] = int,
    nodata_value: Optional[int | float] = None,
) -> np.ndarray:
    
    out_array = data_array
    out_array = utils.replace_nan(data_array=out_array, nodata_value=nodata_value, set_nan=True)
    out_array = np.ma.array(out_array, mask=np.isnan(out_array))
    
    out_array[out_array <= threshold] = 0
    out_array[out_array > threshold] = 1
    
    out_array = out_array.data
    out_array = utils.replace_nan(data_array=out_array, nodata_value=nodata_value, set_value=True)

    return out_array


# Call functions
def _binarize_raster(  # type: ignore[no-any-unimported]
    in_data: rasterio.io.DatasetReader,
    bands: Optional[List[int]] = None,
    thresholds: List[int | float] = List,
    nodata: Optional[List[int | float | None]] = None,
    method: Literal['replace', 'extract'] = 'replace',
) -> Tuple[np.ndarray, dict, dict]:
        raster = in_data
        
        if not bands: bands = list(range(1, raster.count + 1))   
        
        expanded_args = utils.expand_args(selection=bands, nodata=nodata, thresholds=thresholds)
        nodata = expanded_args['nodata']
        thresholds = expanded_args['thresholds']
        
        out_array, out_meta, out_meta_nodata, bands_idx = utils.read_raster(raster=raster, selection=bands, method=method)
        out_settings = {}

        for i, band_idx in enumerate(bands_idx):
            nodata_value = out_meta_nodata[i] if not nodata or nodata[i] is None else nodata[i]
            
            out_array[band_idx] = _binarize_core(data_array=out_array[band_idx],
                                                 threshold=thresholds[i],
                                                 nodata_value=nodata_value)
            
            current_transform = f'transform {band_idx + 1}'
            current_settings = {'band_origin': bands[i],
                                'threshold': thresholds[i],
                                'nodata_meta': out_meta_nodata[i],
                                'nodata_used': nodata_value}
            
            out_settings[current_transform] = current_settings

        return out_array, out_meta, out_settings
    

def _binarize_table(  # type: ignore[no-any-unimported]
    in_data: Union[pd.DataFrame, gpd.GeoDataFrame],
    columns: Optional[List[str]] = None,
    thresholds: List[int | float] = List, 
    nodata: Optional[List[int | float | None]] = None,
) -> Tuple[np.ndarray, dict, dict]:
    dataframe = in_data
    
    out_array, out_column_info, selection = utils.df_to_input_ordered_array(dataframe, columns)
    
    expanded_args = utils.expand_args(selection=selection, nodata=nodata, thresholds=thresholds)
    nodata = expanded_args['nodata']
    thresholds = expanded_args['thresholds']
 
    out_settings = {}
    
    for i, column in enumerate(selection):
        nodata_value = nodata[i] if nodata is not None else None
        
        out_array[i] = _binarize_core(data_array=out_array[i],
                                      threshold=thresholds[i],
                                      nodata_value=nodata_value)

        current_transform = f'transform {i + 1}'
        current_settings = {'original_column_name': column,
                            'original_column_index': dataframe.columns.get_loc(column),
                            'array_index': i,
                            'threshold': thresholds[i],
                            'nodata_used': nodata_value}
        
        out_settings[current_transform] = current_settings

    return out_array, out_column_info, out_settings
 
 
def binarize(  # type: ignore[no-any-unimported]
    in_data: Union[rasterio.io.DatasetReader, pd.DataFrame, gpd.GeoDataFrame],
    selection: Optional[List[int | str]] = None,
    thresholds: List[int | float] = List,
    nodata: Optional[List[int | float | None]] = None,
    method: Optional[Literal['replace', 'extract']] = None,
) -> Tuple[np.ndarray, dict, dict]:
    """Binarize data based on a given threshold.
    
    Replaces values less or equal threshold with 0.
    Replaces values greater than the threshold with 1. 
        
    Takes care of data with NoData values, input can be
    - None
    - user-defined
    If None, NoData will be read from raster metadata.
    If specified, user-input will be preferred.

    Works for multiband raster and multi-column dataframes.
    If no band/column selection specified, all bands/columns will be used.
    
    If only one threshold is specified, it will be used for all (selected) bands.
    If only one NoData value is specified, it will be used for all (selected) bands.
    Contributed parameters will generally be applied for each band/column separately. This way, data can easily be transformed 
    by the same parameters or with different parameters for each band/column (values corresponding to each band/column).

    If method is 'replace', selected bands will be overwritten. Order of bands will not be changed in the output.
    If method is 'extract', only selected bands will be returned. Order in the output corresponds to the order of the specified selection.
    
    Args:
        in_data (rasterio.io.DatasetReader, pd.DataFrame, gpd.GeoDataFrame): Data object to be transformed.
        selection (List[int | str], optional): Bands [int] or columns [str] to be processed. Defaults to None.
        thresholds (List[int | float]): Threshold values for binarization.
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
    valids.append(('Threshold length', parameter.check_parameter_length(selection, thresholds, choice=1)))
    valids.append(('Threshold data type', all(isinstance(item, Union[int, float]) for item in thresholds)))
    valids.append(('NoData length', parameter.check_parameter_length(selection, nodata, choice=1, nodata=True)))  
    
    if nodata is not None: 
        valids.append(('NoData data type', all(isinstance(item, Union[int, float, None]) for item in nodata)))

    if isinstance(in_data, rasterio.DatasetReader):      
        valids.append(('Output method', method == 'replace' or method == 'extract'))

        for item in valids:
            error_msg, validation = item
            
            if validation == False:
                raise InvalidParameterValueException(error_msg)
            
        out_array, out_meta, out_settings = _binarize_raster(in_data=in_data,
                                                             bands=selection,
                                                             thresholds=thresholds,
                                                             nodata=nodata,
                                                             method=method)
        
        return out_array, out_meta, out_settings
    
    if isinstance(in_data, Union[pd.DataFrame, gpd.GeoDataFrame]):
        for item in valids:
            error_msg, validation = item
            
            if validation == False:
                raise InvalidParameterValueException(error_msg)
            
        out_array, out_column_info, out_settings = _binarize_table(in_data=in_data,
                                                                   columns=selection,
                                                                   thresholds=thresholds,
                                                                   nodata=nodata)
        
        return out_array, out_column_info, out_settings
    
    
        