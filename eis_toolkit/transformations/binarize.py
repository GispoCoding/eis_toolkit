import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from typing import Optional, Tuple, Union, List, Literal

from eis_toolkit.exceptions import InvalidParameterValueException, InvalidInputDataException
from eis_toolkit.transformations import utils

# Core functions
def _binarize(  # type: ignore[no-any-unimported]
    data_array: np.ndarray,
    threshold: Union[int, float] = int,
    nodata_value: Optional[int | float] = None,
) -> np.ndarray:
    
    out_array = utils.replace_nan(data_array=data_array, nodata_value=nodata_value, set_nan=True)
    out_array[out_array <= threshold] = 0
    out_array[out_array > threshold] = 1
    out_array = utils.replace_nan(data_array=out_array, nodata_value=nodata_value, set_value=True)

    return out_array


# Call functions
def binarize_raster(  # type: ignore[no-any-unimported]
    raster: rasterio.io.DatasetReader,
    bands: Optional[List[int]] = None,
    tresholds: List[int | float] = List,
    nodata: Optional[List[int | float]] = None,
    method: Literal["in place", "extract"] = str,
) -> np.ndarray:

    if not bands: 
        bands = list(range(0, raster.count))
        
    if not nodata: 
        nodata = raster.nodatavals   
        
    if len(tresholds) == 1 and len(bands) > 1:
        tresholds = tresholds * len(bands)
    
    out_meta = raster.meta
    if method == "in place":
        out_array = raster.read()
    elif method == "extract":
        out_array = raster.read(bands)
        out_meta["count"] = len(bands)
    
    for i in range(0, raster.count):
        nodata_value = raster.nodatavals[i] if not nodata[i] else nodata[i]
            
        if i + 1 == bands[i]:
            out_array[i] = _binarize(data_array=out_array[i],
                                     treshold=tresholds[i],
                                     nodata_value=nodata_value)

    return out_array

# If no bands specified, all bands will be used
# 


# addition for df-nodata handling:
# if not nodata_value: nodata_value = np.nan
# if nodata_replace: nodata_value = np.nan

# file_path = "/path/to/my_file.tif"
# meta_data_dict = {"my_variable": 'my_value'}
# with rasterio.open(file_path, 'r+') as raster:
#     raster.update_tags(**meta_data_dict)