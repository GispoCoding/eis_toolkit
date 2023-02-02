import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from typing import Optional, Tuple, Union, List, Literal

from eis_toolkit.exceptions import InvalidParameterValueException, InvalidInputDataException

def _replace_nan(
    data_array: np.ndarray,
    nodata_value: Optional[int | float] = None,
    set_nan: bool = False,
    set_value: bool = False,
    ) -> np.ndarray:
    
    out_array = data_array
    if not nodata_value: nodata_value = np.nan
    
    if set_nan == True and not np.isnan(nodata_value): out_array[out_array == nodata_value] = np.nan
    elif set_value == True and not np.isnan(nodata_value): out_array[np.isnan(out_array)] = nodata_value
                
    return out_array


def _read_raster(
    raster: rasterio.DatasetReader,
    selection: List[int],
    method: Literal["replace", "extract"],
) -> Tuple[np.ndarray, dict, list, list, list]:
    
    out_meta = raster.meta.copy()
    out_meta_nodata = raster.nodatavals
    bands_idx = [band - 1 for band in selection]

    if method == "replace":
        out_array = raster.read()
    elif method == "extract":
        out_array = raster.read(selection)
        out_meta.update({"count": len(selection)})
        out_meta_nodata = [out_meta_nodata[band] for band in bands_idx]
        bands_idx = list(range(0, len(bands_idx)))
        
    return out_array, out_meta, out_meta_nodata, bands_idx


def _select_columns(
    in_data: Union[pd.DataFrame, gpd.GeoDataFrame],
    columns: Optional[List[str]] = None,
) -> dict:
    """Helper function for selecting relevant columns of a dataframe for transformation.
    
    Works for both pandas as well as geopandas dataframes as following:
    1: Identifies column names of all, numerical, geometry column(s)
    2: Creates an intersection of selected and numerical columns for transformation.
    3: Identifies all unused columns (no transformation)
    
    
    Args:
        in_data (pd.DataFrame, gpd.GeoDataFrame): Dataframe set to be transformed.
        columns (List[str], optional): Columns to be processed. Defaults to None.

    Returns:
        out_dict (dict): Dictionary with column information.

    Raises:
        InvalidParameterValueException
        InvalidInputDataException
    """
    
    columns_all = in_data.columns.to_list()
    columns_numeric = in_data.select_dtypes(include="number").columns.to_list()
    columns_geometry = in_data.select_dtypes(include="geometry").columns.to_list()
    
    if columns is not None:       
        columns_transform = [column for column in columns_numeric if column in columns]
    else:
        columns_transform = columns_numeric
    
    if not columns_transform:
      if not columns_numeric:
        raise InvalidInputDataException
      else:
        raise InvalidParameterValueException
    
    columns_no_transform = [column for column in columns_all if not column in columns_transform]
    if columns_geometry: columns_no_transform.remove(columns_geometry)
    
    out_dict = {"columns_all": columns_all,
                "columns_numeric": columns_numeric,
                "columns_geometry": columns_geometry,
                "columns_transform": columns_transform,
                "columns_no_transform": columns_no_transform}
    
    return out_dict