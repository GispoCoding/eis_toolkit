import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from typing import Optional, Tuple, Union, List, Literal

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.transformations import utils

# Core functions
def _binarize_core(  # type: ignore[no-any-unimported]
    data_array: np.ndarray,
    threshold: Union[int, float] = int,
    nodata_value: Optional[int | float] = None,
) -> np.ndarray:
    
    out_array = data_array
    out_array = utils._replace_nan(data_array=out_array, nodata_value=nodata_value, set_nan=True)
    out_array = np.ma.array(out_array, mask=np.isnan(out_array))
    
    out_array[out_array <= threshold] = 0
    out_array[out_array > threshold] = 1
    
    out_array = out_array.data
    out_array = utils._replace_nan(data_array=out_array, nodata_value=nodata_value, set_value=True)

    return out_array


# Call functions
def _binarize_raster(  # type: ignore[no-any-unimported]
    in_data: rasterio.io.DatasetReader,
    bands: Optional[List[int]] = None,
    thresholds: List[int | float] = List,
    nodata: Optional[List[int | float | None]] = None,
    method: Literal["replace", "extract"] = str,
) -> Tuple[np.ndarray, dict, dict]:
        raster = in_data
        
        if not bands: bands = list(range(1, raster.count + 1))    
        if method == "replace" and len(thresholds) == 1 and len(bands) > 1: thresholds = thresholds * raster.count      
        if method == "extract" and len(thresholds) == 1 and len(bands) > 1: thresholds = thresholds * len(bands)   
        
        out_array, out_meta, out_meta_nodata, bands_idx = utils._read_raster(raster=raster, selection=bands, method=method)
        out_settings = {}

        for i, band_idx in enumerate(bands_idx):
            if nodata and len(nodata) == 1: nodata = nodata * len(bands_idx)
            nodata_value = out_meta_nodata[i] if not nodata or not nodata[i] else nodata[i]
            
            out_array[band_idx] = _binarize_core(data_array=out_array[band_idx],
                                                 threshold=thresholds[i],
                                                 nodata_value=nodata_value)
            
            current_band = f"band {band_idx + 1}"
            current_settings = {"band_origin": bands[i],
                                "threshold": thresholds[i],
                                "nodata_meta": out_meta_nodata[i],
                                "nodata_used": nodata_value}
            out_settings[current_band] = current_settings

        return out_array, out_meta, out_settings
    
    
def binarize(  # type: ignore[no-any-unimported]
    in_data: Union[rasterio.io.DatasetReader, pd.DataFrame, gpd.GeoDataFrame],
    selection: Optional[List[int]] = None,
    thresholds: List[int | float] = List,
    nodata: Optional[List[int | float | None]] = None,
    method: Literal["replace", "extract"] = str,
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
    
    If only one threshold is specified, it will be used for all selected bands.
    If only one NoData value is specified, it will be used for all selected bands.
    Contributed parameters will generally be applied for each band/column separately.
    
    If method is 'replace', selected bands/colums will be overwritten. Order of bands will not be changed in the output.
    If method is 'extract', only selected bands/columns will be returned. Order of bands in the output corresponds to the order of selected bands in the input.
    
    Args:
        in_data (rasterio.io.DatasetReader, pd.DataFrame, gpd.GeoDataFrame): Data object to be transformed.
        selection (List[int], optional): Band or column indicies to be processed. Defaults to None.
        thresholds (List[int | float]): Threshold values for binariziation.
        nodata (List[int | float], optional): NoData values to be considered. Defaults to None.
        method (Literal): Switch for data output.

    Returns:
        out_array (np.ndarray): The transformed data.
        out_meta (dict): Updated metadata with new band count.
        out_settings (dict): Return of the input settings related to the new ordered output.

    Raises:
        InvalidParameterValueException: The input contains invalid values.
    """    

    if thresholds is not None:   
        if not all(isinstance(item, Union[int, float, None]) for item in thresholds):
            raise InvalidParameterValueException
    else:
        raise InvalidParameterValueException
    
    if nodata is not None:
        if not all(isinstance(item, Union[int, float, None]) for item in nodata):
            raise InvalidParameterValueException
    
    if selection is not None:
        if not all(isinstance(item, int) for item in selection):
            raise InvalidParameterValueException
        elif len(set(selection)) != len(selection):
            raise InvalidParameterValueException
        elif len(selection) < len(thresholds):
            raise InvalidParameterValueException
        elif (len(selection) > 1 and len(thresholds) > 1) and (len(selection) != len(thresholds)):
            raise InvalidParameterValueException
             
    if selection is not None and nodata is not None:
        if (len(nodata) > 1 and len(selection) > 1) and (len(nodata) != len(selection)):
            raise InvalidParameterValueException
    
    
    if isinstance(in_data, rasterio.DatasetReader):       
        if selection is not None:
            if max(selection) > in_data.count:
                raise InvalidParameterValueException
            elif len(selection) > in_data.count:
                raise InvalidParameterValueException
        
        if method != "replace" and method != "extract":
            raise InvalidParameterValueException

        out_array, out_meta, out_settings = _binarize_raster(in_data=in_data,
                                                             bands=selection,
                                                             thresholds=thresholds,
                                                             nodata=nodata,
                                                             method=method)
        
        return out_array, out_meta, out_settings