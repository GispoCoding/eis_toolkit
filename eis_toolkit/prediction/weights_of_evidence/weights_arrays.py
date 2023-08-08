
from typing import Tuple, List
import pandas as pd
import numpy as np
import rasterio
import functools


def _raster_array(
    ev_rst: rasterio.io.DatasetReader,
    df_wgts_nan: pd.DataFrame, col: str
) -> np.ndarray:
    #rstr_meta = ev_rst.meta.copy()
    rstr_arry = np.array(ev_rst.read(1))
    s = rstr_arry.shape
    #print(df_wgts_nan.Class)
    wgts_mapping_dct = {}
    wgts_mapping_dct = pd.Series(df_wgts_nan.loc[:, col],index=df_wgts_nan.Class).to_dict() 
    replace_array = np.array([list(wgts_mapping_dct.keys()), list(wgts_mapping_dct.values())]) 
    rstr_arry_wgts = rstr_arry.reshape(-1)
    mask_array = np.isin(rstr_arry_wgts, replace_array[0, :])
    ss_rplc_array = np.searchsorted(replace_array[0, :], rstr_arry_wgts[mask_array])
    rstr_arry_rplcd = replace_array[1, ss_rplc_array]
    rstr_arry_rplcd = rstr_arry_rplcd.reshape(s)
    return rstr_arry_rplcd


def raster_array(
    ev_rst: rasterio.io.DatasetReader,
    df_wgts_nan: pd.DataFrame, col: str
) -> np.ndarray:
    """Converts the generalized weights dataaframe to numpy arrays with the extent and shape of the input raster

    Args:
        ev_rst (rasterio.io.DatasetReader): The evidential raster with spatial resolution and extent identical to that of the dep_rst.
        df_wgts_nan (pd.DataFrame): Generalized weights dataframe with info on NaN data also.
        col (str): Columns to use for generation of raster object arrays.

    Returns:
        np.ndarray: Individual raster object arrays for generalized or unique classes, generalized weights and standard deviation of generalized weights
    """
    raster_array_ = _raster_array(ev_rst, df_wgts_nan, col)
    return raster_array_


def _weights_arrays(
    ev_rst: rasterio.io.DatasetReader,
    df_wgts: pd.DataFrame, 
    col_names: List
) -> Tuple[List, dict]:
    rstr_meta = ev_rst.meta.copy()
    list_cols = list(df_wgts.columns)
    nan_row = {val: -1.e+09 for val in list_cols}
    nan_row_df = pd.DataFrame.from_dict(nan_row, orient = 'index')
    nan_row_df_t = nan_row_df.T
    df_wgts_nan = pd.concat([nan_row_df_t, df_wgts])
    class_rstr, w_gen_rstr, std_rstr = map(
        functools.partial(raster_array, ev_rst, df_wgts_nan), 
        col_names)
    gen_arrys = [class_rstr, w_gen_rstr, std_rstr]
    return gen_arrys, rstr_meta

def weights_arrays(
    ev_rst: rasterio.io.DatasetReader,
    df_wgts: pd.DataFrame, col_names: List
) -> Tuple[List, dict]:
    """Calls the raster_arrays function to convert the generalized weights dataaframe to numpy arrays. 
    
    Args:
        ev_rst (rasterio.io.DatasetReader): The evidential raster with spatial resolution and extent identical to that of the dep_rst.
        df_wgts (pd.DataFrame): Dataframe with the weights.
        col_names (List): Columns to generate the arrays from.

    Returns:
        gen_arrys (List): List of individual raster object arrays for generalized or unique classes, generalized weights and standard deviation of generalized weights
        rstr_meta (dict): Raster array's metadata.
    """
  
    gen_arrys, rstr_meta = _weights_arrays(ev_rst, df_wgts, col_names)
    return gen_arrys, rstr_meta
