import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from typing import Optional, Tuple, Union, List, Literal

from eis_toolkit.exceptions import InvalidParameterValueException, InvalidInputDataException
from eis_toolkit.transformations import utils
from eis_toolkit.checks import parameter


# Core functions
def _log_transform_core(  # type: ignore[no-any-unimported]
    data_array: np.ndarray,
    base: int,
    nodata_value: Optional[int | float] = None,
) -> np.ndarray:
    
    out_array = utils.replace_nan(data_array=data_array, nodata_value=nodata_value, set_nan=True)
    out_array[out_array <= 0] = np.nan
    out_array[np.isinf(out_array)] = np.nan

    if base == 2: out_array = np.log2(out_array)
    if base == 10: out_array = np.log10(out_array)

    out_array = utils.replace_nan(data_array=out_array, nodata_value=nodata_value, set_value=True)

    return out_array


# Call functions
def _log_transform_raster(  # type: ignore[no-any-unimported]
    in_data: rasterio.io.DatasetReader,
    bands: Optional[List[int]] = None,
    base: List[int] = [2],
    nodata: Optional[List[int | float | None]] = None,
    method: Literal["replace", "extract"] = "replace",
) -> Tuple[np.ndarray, dict, dict]:
        raster = in_data
        
        if not bands: bands = list(range(1, raster.count + 1))    
        
        expanded_args = utils.expand_args(selection=bands, nodata=nodata, base=base)
        nodata = expanded_args["nodata"]
        base = expanded_args["base"]
         
        out_array, out_meta, out_meta_nodata, bands_idx = utils.read_raster(raster=raster, selection=bands, method=method)
        out_settings = {}

        for i, band_idx in enumerate(bands_idx):
            nodata_value = out_meta_nodata[i] if not nodata or nodata[i] is None else nodata[i]
            
            out_array[band_idx] = _log_transform_core(data_array=out_array[band_idx],
                                                      base=base[i],
                                                      nodata_value=nodata_value)
            
            current_band = f"band {band_idx + 1}"
            current_settings = {"band_origin": bands[i],
                                "base": base[i],
                                "nodata_meta": out_meta_nodata[i],
                                "nodata_used": nodata_value}
            out_settings[current_band] = current_settings

        return out_array, out_meta, out_settings
    
    
def log_transform(  # type: ignore[no-any-unimported]
    in_data: Union[rasterio.io.DatasetReader, pd.DataFrame, gpd.GeoDataFrame],
    selection: Optional[List[int]] = None,
    base: List[int] = [2],
    nodata: Optional[List[int | float | None]] = None,
    method: Literal["replace", "extract"] = "replace",
) -> Tuple[np.ndarray, dict, dict]:
    """Logarithmic transformation.
    
    Transforms input data with log2 or log10.
        
    Takes care of data with NoData values, input can be
    - None
    - user-defined
    If None, NoData will be read from raster metadata.
    If specified, user-input will be preferred.
    
    If infinity values occur, they will be replaced by NaN.
    If values <= 0 occur, they will be replaced by NaN.
    
    Works for multiband raster and multi-column dataframes.
    If no band/column selection specified, all bands/columns will be used.
    
    If only one base-value is specified, it will be used for all (selected) bands.
    If only one NoData value is specified, it will be used for all (selected) bands.
    Contributed parameters will generally be applied for each band/column separately. This way, data can easily be transformed 
    by the same parameters or with different parameters for each band/column (values corresponding to each band/column).

    If method is 'replace', selected bands/colums will be overwritten. Order of bands will not be changed in the output.
    If method is 'extract', only selected bands/columns will be returned. Order in the output corresponds to the order of the specified selection.
    
    Args:
        in_data (rasterio.io.DatasetReader, pd.DataFrame, gpd.GeoDataFrame): Data object to be transformed.
        selection (List[int], optional): Bands or columns to be processed. Defaults to None.
        base (List[int]): Log-base to be applied. Possible values are 2 and 10. Defaults to 2.
        nodata (List[int | float], optional): NoData values to be considered. Defaults to None.
        method (Literal["replace", "extract"]): Switch for data output. Defaults to "replace".

    Returns:
        out_array (np.ndarray): The transformed data.
        out_meta (dict): Updated metadata with new band count.
        out_settings (dict): Return of the input settings related to the new ordered output.

    Raises:
        InvalidParameterValueException: The input contains invalid values.
    """    
    valids = parameter.check_band_selection(in_data, selection)
    valids.append(("Base length", parameter.check_parameter_length(selection, base, choice=1)))
    valids.append(("NoData length", parameter.check_parameter_length(selection, nodata, choice=1, nodata=True)))    
    valids.append(("Base data type", all(isinstance(item, int) for item in base)))
    valids.append(("Base value", all([item == 2 or item == 10 for item in base])))
    
    if nodata is not None: 
        valids.append(("NoData data type", all(isinstance(item, Union[int, float, None]) for item in nodata)))
        
    if isinstance(in_data, rasterio.DatasetReader):       
        valids.append(("Output method", method == "replace" or method == "extract"))

        for item in valids:
            error_msg, validation = item
            
            if validation == False:
                raise InvalidParameterValueException(error_msg)

        out_array, out_meta, out_settings = _log_transform_raster(in_data=in_data,
                                                                  bands=selection,
                                                                  base=base,
                                                                  nodata=nodata,
                                                                  method=method)
        
        return out_array, out_meta, out_settings