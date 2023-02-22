import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from typing import Optional, Tuple, Union, List, Literal

from eis_toolkit.exceptions import InvalidParameterValueException, InvalidInputDataException


def expand_args(
    selection: List[int | str],
    **kwargs
) -> dict:
    
  out_args = kwargs

  for key, value in out_args.items():
    if value is not None and len(value) == 1: 
        out_args[key] = out_args[key] * len(selection)

  return out_args


def replace_nan(
    data_array: np.ndarray,
    nodata_value: Optional[int | float] = None,
    set_nan: bool = False,
    set_value: bool = False,
) -> np.ndarray:
    
    out_array = data_array
    if nodata_value is None: nodata_value = np.nan
    
    if set_nan == True and not np.isnan(nodata_value): out_array[out_array == nodata_value] = np.nan
    elif set_value == True and not np.isnan(nodata_value): out_array[np.isnan(out_array)] = nodata_value
                
    return out_array


def read_raster(
    raster: rasterio.DatasetReader,
    selection: List[int],
    method: Literal['replace', 'extract'],
) -> Tuple[np.ndarray, dict, list, list]:
    
    out_meta = raster.meta.copy()
    out_meta_nodata = raster.nodatavals
    bands_idx = [band - 1 for band in selection]

    if method == 'replace':
        out_array = raster.read()
    elif method == 'extract':
        out_array = raster.read(selection)
        out_meta.update({'count': len(selection)})
        out_meta_nodata = [out_meta_nodata[band] for band in bands_idx]
        bands_idx = list(range(0, len(bands_idx)))
        
    return out_array, out_meta, out_meta_nodata, bands_idx


def select_columns(
    in_data: Union[pd.DataFrame, gpd.GeoDataFrame],
    columns: Optional[List[str]] = None,
) -> dict:
    """Helper function for selecting relevant columns of a dataframe for transformation.
    
    Works for both pandas as well as geopandas dataframes as following:
    1: Identifies column names of all, numerical, geometry column(s)
    2: Creates an intersection of selected and numerical columns for transformation.
    3: Identifies all columns not appropriate for transformation (no transformation)
    
    Args:
        in_data (pd.DataFrame, gpd.GeoDataFrame): Dataframe set to be transformed.
        columns (List[str], optional): Selection of columns to be processed. Defaults to None.

    Returns:
        out_dict (dict): Dictionary with column information.
    """
    
    columns_all = in_data.columns.to_list()
    columns_numeric = in_data.select_dtypes(include='number').columns.to_list()
    columns_geometry = in_data.select_dtypes(include='geometry').columns.to_list()
    
    if columns is not None:       
        columns_transform = [column for column in columns_numeric if column in columns]
    else:
        columns_transform = columns_numeric
    
    columns_no_transform = [column for column in columns_all if not column in columns_transform]
    if columns_geometry: columns_no_transform.remove(columns_geometry)
    
    out_dict = {'columns_all': columns_all,
                'columns_numeric': columns_numeric,
                'columns_geometry': columns_geometry,
                'columns_transform': columns_transform,
                'columns_no_transform': columns_no_transform}
    
    return out_dict


def df_to_input_ordered_array(
    in_data: Union[pd.DataFrame, gpd.GeoDataFrame],
    columns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, dict, list]:
    """Helper function for ordering selection and converting a dataframe to a numpy array.
    
    Works for both pandas as well as geopandas dataframes as following:
    1: Identifies column names of all, numerical, geometry column(s)
    2: Creates an intersection of selected and numerical columns for transformation.
    3: Identifies all columns not appropriate for transformation (no transformation)
    
    Args:
        in_data (pd.DataFrame, gpd.GeoDataFrame): Dataframe set to be transformed.
        columns (List[str], optional): Selection of columns to be processed. Defaults to None.

    Returns:
        out_array (np.ndarray): Numpy array.
        out_dict (dict): Dictionary with columns classified by use-case
        out_selection (list): Ordered selection based on the column-order
    """
    out_dict = select_columns(in_data, columns)
    out_selection = out_dict['columns_transform']
    
    if columns is not None:
        out_selection = sorted(out_selection, key=lambda x: columns.index(x))
    
    out_array = in_data.loc[:, out_selection].values.T
    
    return out_array, out_dict, out_selection
    