"""Basic calculations"""

import pandas as pd
import rasterio
import numpy as np


def _basic_calculations(
    ev_rst: rasterio.io.DatasetReader,
    dep_rst: rasterio.io.DatasetReader,
    nan_val: float
) -> pd.DataFrame:

    geol, dep_ar = np.array(ev_rst.read(1)), np.array(dep_rst.read(1))
    tot_pxls = np.size(geol) - np.count_nonzero(geol <= nan_val)
    dep1s, dep0s = np.count_nonzero(dep_ar == 1), np.count_nonzero(dep_ar == 0)
    d_flat, g_flat = dep_ar.flatten(), geol.flatten()
    df_flt = pd.DataFrame({"Clss": g_flat, "Ds": d_flat})
    Geol_Dep = df_flt.groupby("Clss")["Ds"]
    Geol_Unq = np.unique(g_flat)
    Geol_Cnt, Geol_Dep_Sum = Geol_Dep.count(), Geol_Dep.sum()
    Geol_NoDep = Geol_Dep.count() - Geol_Dep.sum()

    calc_df = pd.DataFrame(
        {"Class": Geol_Unq, "Count": Geol_Cnt,
         "Point_Count": Geol_Dep_Sum,
         "No_Dep_Cnt": Geol_NoDep, "Total_Area": tot_pxls,
         "Total_Deposits": dep1s, "Tot_No_Dep_Cnt": dep0s
         }
    )

    cols = ['Class', 'Count', 'Point_Count', 'No_Dep_Cnt',
            'Total_Area', 'Total_Deposits', 'Tot_No_Dep_Cnt']
    # replace_cols = ['Class', 'Count', 'Point_Count', 'No_Dep_Cnt', 'Total_Area',
    # 'Total_Deposits', 'Tot_No_Dep_Cnt', 'Dep_outsidefeat', 'Non_Feat_Non_Dep']

    return (calc_df
            [cols]
            .assign(Dep_outsidefeat=lambda calc_df: calc_df.Total_Deposits - calc_df.Point_Count)
            .assign(Non_feat_Pxls=lambda calc_df: calc_df.Total_Area - calc_df.Count)
            .assign(Non_Feat_Non_Dep=lambda calc_df: calc_df.Non_feat_Pxls - calc_df.Dep_outsidefeat)
            # [replace_cols].replace({0: 0.0001, 1: 1.0001}), #FIX THIS?
            )


def basic_calculations(
    ev_rst: rasterio.io.DatasetReader,
    dep_rst: rasterio.io.DatasetReader,
    nan_val: float
) -> pd.DataFrame:
    """Performs basic calculations about the number of point pixels per class of the input raster.

    Args:
        ev_rst (rasterio.io.DatasetReader): The evidential raster.
        dep_rst (rasterio.io.DatasetReader): Deposit raster
        nan_val (float): value of no data.

    Returns:
        basic_clcs (pandas.DataFrame): dataframe with basic calculations.

    """
    basic_clcs = _basic_calculations(ev_rst, dep_rst, nan_val)
    return basic_clcs
