import numpy as np
import functools

from eis_toolkit import exceptions
from typing import Dict, List, Tuple, Literal

import pandas as pd
import rasterio


SMALL_VALUE = 0.0001
LARGE_VALUE = 1.0001



def weights_of_evidence(
    evidential_raster: rasterio.io.DatasetReader,
    deposit_raster: rasterio.io.DatasetReader,
    weights_type: Literal['unique', 'ascending', 'descending'] = 'unique',
    studentized_contrast: float = 2,
) -> Tuple[pd.DataFrame, List, Dict]:
    """Calculates weights of spatial associations.

    Args:
        evidential_raster: The evidential raster with spatial resolution and extent identical to that of the deposit_raster.
        deposit_raster: Raster representing the mineral deposits or occurences point data.
        weights_type: Accepted values are 'unique' for unique weights, 'ascending' for cumulative ascending weights,
            'descending' for cumulative descending weights. Defaults to 'unique'.
        studentized_contrast: Studentized contrast value to be used for genralization of classes. 
            Not needed if weights_type is 'unique'. Defaults to 2.

    Returns:
        weights_df: Dataframe with weights of spatial association between the input rasters
        raster_gen: List of output raster arrays with generalized or unique classes, generalized weights
            and standard deviation of generalized weights
        raster_meta: Raster array's metadata.

    Raises:
        ValueError: Accepted values of weights_type are 0, 1, 2 for unique, cumulative ascending and cumulative descending weights respectively.
        The below exceptions will be incorporated into the function later as the development for other related functions progresses in the toolkit.
        NonMatchingCrsException: The input rasters are not in the same crs
        InvalidParameterValueException: Accepted values of weights_type are 0, 1, 2 for unique, cumulative ascending and cumulative descending weights respectively. (status - pending)
        NonMatchingTransformException: The input rasters do not have the same cell size and/or same extent (status - pending)
        NonMatchingCoRegistrationException: The input rasters are not coregistered (status - pending)
    """

    basic_calculations_df = _basic_calculations(evidential_raster, deposit_raster)
    weights_df, raster_gen, raster_meta = _weights_calculations(
        evidential_raster,
        basic_calculations_df,
        weights_type, 
        studentized_contrast
    )
    return weights_df, raster_gen, raster_meta


def _basic_calculations(
    evidential_raster: rasterio.io.DatasetReader, deposit_raster: rasterio.io.DatasetReader
) -> pd.DataFrame:
    """Performs basic calculations about the number of point pixels per class of the input raster.

    Args:
        evidential_raster (rasterio.io.DatasetReader): The evidential raster.
        deposit_raster (rasterio.io.DatasetReader): Deposit raster.

    Returns:
        basic_calculations_df (pandas.DataFrame): dataframe with basic calculations.
    """

    # Read raster data
    evidential_array, deposit_array = np.array(evidential_raster.read(1)), np.array(deposit_raster.read(1))

    # Convert nodata values to np.nan
    evidential_array[evidential_array == evidential_raster.meta.nodata] = np.nan
    deposit_array[deposit_array == deposit_raster.meta.nodata] = np.nan

    total_pixels = np.size(evidential_array) - np.isnan(evidential_array).sum()
    dep1s, dep0s = np.count_nonzero(deposit_array == 1), np.count_nonzero(deposit_array == 0)  # CHECK

    df_flat = pd.DataFrame(
        {"Class": evidential_array.flatten(),
         "Deposits": deposit_array.flatten()}
    )

    geol_dep = df_flat.groupby("Class")["Deposits"]
    geol_count = geol_dep.count()
    geol_dep_sum = geol_dep.sum()

    basic_calculations_df = pd.DataFrame(
        {
            "Class": np.unique(evidential_array),
            "Count": geol_count,
            "Point_Count": geol_dep_sum,
            "No_Dep_Cnt": geol_count - geol_dep_sum,
            "Total_Area": total_pixels,
            "Total_Deposits": dep1s,
            "Tot_No_Dep_Cnt": dep0s,
            'Dep_outsidefeat': dep1s - geol_dep_sum,
            'Non_feat_pixels': total_pixels - geol_count,
        }
    )
    basic_calculations_df['Non_Feat_Non_Dep'] = basic_calculations_df['Non_feat_pixels'] - basic_calculations_df['Dep_outsidefeat']

    return basic_calculations_df


def _weights_calculations(
    evidential_raster: rasterio.io.DatasetReader,
    basic_calculations_df: pd.DataFrame,
    weight_type: Literal['unique', 'ascending', 'descending'],
    studentized_contrast: float
) -> Tuple[pd.DataFrame, List, dict]:
    """Calculates weights of spatial associations.

    Args:
        evidential_raster: The evidential raster.
        basic_calculations_df: Dataframe obtained from basic_calculations function.
        weights_type: Accepted values are 'unique' for unique weights, 'ascending' for cumulative ascending weights,
            'descending' for cumulative descending weights.
        studentized_contrast: Studentized contrast value to be used for genralization of classes.
            Not needed if weight_type = 'unique'.

    Returns:
        weights_df (pd.DataFrame): Dataframe with weights of spatial association between the input rasters.
        gen_arrays (List): List of output raster arrays with generalized or unique classes, generalized weights and standard deviation of generalized weights.
        raster_meta (dict): Raster array's metadata.

    """
    weights_df_test = _weights_type(basic_calculations_df, weight_type)
    wpls_df = _positive_weights(weights_df_test)
    wmns_df = _negative_weights(wpls_df, weight_type)
    contrast_df = _contrast(wmns_df)

    if weight_type == 'unique':
        cat_wgts = _weights_cleanup(contrast_df)
        col_names = ["Class", "WPlus", "S_WPlus"]
        gen_arrys, raster_meta = _weights_arrays(evidential_raster, cat_wgts, col_names)
        return cat_wgts, gen_arrys, raster_meta
    else:
        num_weights = _weights_generalization(
            contrast_df,
            weight_type,
            studentized_contrast,
        )
        col_names = ["Gen_Class", "Gen_Weights", "S_Gen_Weights"]
        gen_arrys, raster_meta = _weights_arrays(evidential_raster, num_weights, col_names)
        return num_weights, gen_arrys, raster_meta



def _weights_type(df: pd.DataFrame, weight_type: Literal['unique', 'ascending', 'descending']) -> pd.DataFrame:
    """
    Based on the type of weights selected by the user, the function performs the sorting and cumulative count calculations.
    
    Args:
        df (pandas.DataFrame): The dataframe with basic calculations performed in the basic_calculations function
        weight_type(str): 'unique' = unique weights, 'ascending' = cumulative ascending weights,
            'descending' = cumulative descending weights

    Returns:
        df_data: The dataframe with data values sorted if weights calculation type is numerical
            (i.e., weight_type = 'ascending' or 'descending')
    """
    # To replace: Point_Count, No_Dep_Cnt
    df_data = df.dropna(subset=['Class'])

    if weight_type != 'unique':
        df_data = df_data.copy()  # Avoid SettingWithCopyWarning
        df_data.rename(columns={"Point_Count": "Act_Point_Count"}, inplace=True)
        df_data.sort_values(by="Class", ascending=(weight_type == 'ascending'), inplace=True)
        df_data['Point_Count'] = df_data['Act_Point_Count'].cumsum()
        df_data['cmltv_cnt'] = df_data['Count'].cumsum()
        df_data['No_Dep_Cnt'] = df_data['No_Dep_Cnt'].cumsum()
    
    return df_data


def _positive_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates positive weights of spatial associations between the input data and the points.
    
    Args:
        df: The dataframe containing data values, obtained from the weights_type function
    Returns:
        df: The dataframe with positive weights of spatial associaton for each data class
    """
    # To replace: Point_Count, Num_wpls (also LARGE)
    df['Deno_wpls'] = df['No_Dep_Cnt'] / df['Tot_No_Dep_Cnt']
    df['wpls'] = np.log(df['Num_wpls']) - np.log(df['Deno_wpls'])
    df['var_wpls'] = (1 / df['Point_Count']) + (1 / df['No_Dep_Cnt'])
    df['s_wpls'] = np.sqrt(df['var_wpls'])
    return df


def _negative_weights(df_: pd.DataFrame, weight_type: Literal['unique', 'ascending', 'descending']) -> pd.DataFrame:
    """Calculates negative weights of spatial associations between the input data and the points.
    
    Args:
        df: The dataframe containing already the positive weights for the data values.
        weight_type: 'unique' = unique weights, 'ascending' = cumulative ascending weights,
            'descending' = cumulative descending weights
    Returns:
        df_wmns: The dataframe with the negative weights of spatial association for each class.
    """
    # To replace: Num_wmns, Dep_outsidefeat, Non_Feat_Pxls_cm, Non_Feat_Non_Dep, Deno_wmns
    df = df_.copy()
    # if weight_type != 0:
    #     df.rename(columns={"Count": "Act_Count", "cmltv_cnt": "Count"}, inplace=True)

    df["Num_wmns"] = df['Total_Deposits'] - (df['Point_Count'] / df['Total_Deposits'])
    df['Dep_outsidefeat'] = df['Total_Deposits'] - df['Point_Count']
    df['Non_Feat_Pxls_cm'] = df['Total_Area'] - df['Count']
    df['Non_Feat_Non_Dep'] = df['Non_Feat_Pxls_cm'] - df['Dep_outsidefeat']
    df['Deno_wmns'] = df['Non_Feat_Pxls_cm'] - (df['Dep_outsidefeat'] / df['Tot_No_Dep_Cnt'])
    df['wmns'] = np.log(df['Num_wmns']) - np.log(df['Deno_wmns'])
    df['var_wmns'] = (1 / df['Dep_outsidefeat']) + (1 / df['Non_Feat_Non_Dep'])
    df['s_wmns'] = np.sqrt(df['var_wmns'])

    df = df.round(4)

    df.loc[df["Num_wmns"] == df["Deno_wmns"], "Num_wmns"] = LARGE_VALUE
    return df


def _contrast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the contrast and the studentized contrast values from the postive and negative spatial
    associations quantified as weights in the positive_weights and negative_weights functions.

    Args:
        df: The dataframe containing the positive and negative weights of spatial associations.
    Returns:
        Dataframe with contrast and studentized contrast values.
    """
    df['contrast'] = df['wpls'] - df['wmns']
    df['s_contrast'] = np.sqrt(df['var_wpls'] + df['var_wmns'])
    df['Stud_Cont'] = df['contrast'] / df['s_contrast']

    df = df.round(3)
    return df










def _weights_cleanup(df: pd.DataFrame, weight_type: int = 0) -> pd.DataFrame:
    """
    Removes unnecessary columns and creates a clean dataframe with important spatial associations quantities.
    For caterogical data with weights calculations type 'unique', this function is called after the weights calculations.
    For numerical data this function is called after reclassification and generalized weights calculations.

    Args:
        df: Dataframe with all the calculations; obtained from the contrast function (for categorical data) or from the generalized weights function (for ordinal data)
        weight_type: 
    Returns:
        Final dataframe with only the necessary values.
    """

    drop_cols = [
        "No_Dep_Cnt",
        "Total_Area",
        "Total_Deposits",
        "Tot_No_Dep_Cnt",
        "Dep_outsidefeat",
        "Non_feat_pixels",
        "Non_Feat_Non_Dep",
        "N_cls",
        "Num_wpls",
        "Deno_wpls",
        "var_wpls",
        "Num_wmns",
        "Non_Feat_Pxls_cm",
        "Deno_wmns",
        "var_wmns",
        "var_wpls_gen",
        "cmltv_dep_cnt",
        "cmltv_cnt",
        "cmltv_no_dep_cnt",
    ]

    if weight_type != 0:
        cols_rename = {
            "Class": "Class",
            "Point_Count": "Cmltv. Point Count",
            "Count": "Cmltv. Count",
            "Act_Count": "Count_",
            "Act_Point_Count": "Point Count_",
            "wpls": "WPlus",
            "s_wpls": "S_WPlus",
            "wmns": "WMinus",
            "s_wmns": "S_WMinus",
            "contrast": "Contrast",
            "s_contrast": "S_Contrast",
            "Stud_Cont": "Stud. Contrast",
            "Rcls": "Gen_Class",
            "W_Gen": "Gen_Weights",
            "s_wpls_gen": "S_Gen_Weights",
        }
    else:
        cols_rename = {
            "Class": "Class",
            "Point_Count": "Point Count",
            "Count": "Count",
            "wpls": "WPlus",
            "s_wpls": "S_WPlus",
            "wmns": "WMinus",
            "s_wmns": "S_WMinus",
            "contrast": "Contrast",
            "s_contrast": "S_Contrast",
            "Stud_Cont": "Stud. Contrast",
        }
    df = df.drop([col for col in drop_cols if col in df.columns], axis=1).rename(columns=cols_rename).round(4)
    return df


def _weights_generalization(weights_df: pd.DataFrame, weight_type: int, studentized_contrast: float = 2) -> pd.DataFrame:
    """Identifies the favourable and unfavorable classes based on the weights for ascending and descending weights and recalculates the generalized weitghts
    Args:
        weights_df (pandas.DataFrame): dataframe with the weights
        studentized_contrast (float, def = 2): studentized contrast value to be used for genralization of classes
    Returns:
        wgts_fnl (pandas.DataFrame): dataframe with generalized weights and generalized classes
    """

    rcls_df = _reclass_gen(weights_df, studentized_contrast)
    wgts_gen = gen_weights_finalization(rcls_df)
    wgts_fnl = _weights_cleanup(wgts_gen, weight_type)
    return wgts_fnl


def _reclass_gen(df: pd.DataFrame, studentized_contrast: float = 2) -> pd.DataFrame:
    """Performs reclassification of classes into favourable and unfavourable categories for ordianl data, based on studentized contrast values provided by the user.
    Args:
        df (pandas.DataFrame): The dataframe with all the weights calculations; obtained from the contrast function
        studentized_contrast (float, def = 2): The threshold for studentized contrast for reclassification
    Returns:
        df_rcls (pandas.DataFrame): The dataframe with data classes categorized as favourable and unfavourable
    Raises:
        UnFavorableClassDoesntExistException: Failure to generalize classes using the given studentised contrast threshold value. Class 1 (unfavorable class) doesn't exist
        FavorableClassDoesntExistException: Failure to generalize classes using the given studentised contrast threshold value. Class 2 (favorable class) doesn't exist
    """

    df["Rcls"] = df.Stud_Cont.ge(studentized_contrast)[::-1].cummax() + 1

    df_studentized_contrast = df[["Class", "contrast", "Stud_Cont", "Rcls"]].round(4)

    if 1 not in df["Rcls"].values:
        print(df_studentized_contrast)
        raise exceptions.UnFavorableClassDoesntExistException(
            """Exception error: Class doesn't exist. 
    For the given studentized contrast value, none of the classes were classified to the 'Unfavorable' class, i.e., class 1. 
    Check the displayed studentized contrast values ('Stud_Cont') and run weights calculations again using a suitable threshold value."""
        )

    elif 2 not in df["Rcls"].values:
        print(df_studentized_contrast)
        raise exceptions.FavorableClassDoesntExistException(
            """Exception error: Class doesn't exist. 
        For the given studentized contrast value none of the classes were classified to the 'Favorable' class, i.e., class 2. 
        Check the displayed studentized contrast values ('Stud_Cont') and run weights calculations again using a suitable threshold value."""
        )

    else:
        df_ = df.round(4).sort_values(by="Class", ascending=True)
        return df_
    

def gen_weights_finalization(df: pd.DataFrame) -> pd.DataFrame:
    """Calls the gen_weights function and calculates positives weights of associations for each generalized class

    Args:
        df (pd.DataFrame): Dataframe with weights of associations
    Returns:
        df (pd.DataFrame): Dataframe with positives weights of associations for generalized classes
    """
    gen_cls = [1, 2]
    gw_1, gw_2 = map(functools.partial(gen_weights, df), gen_cls)
    gw = pd.concat([gw_1, gw_2])
    for i, clss in enumerate(gen_cls):
        w = gw.iloc[i, 31]
        s_w = gw.iloc[i, 33]
        v_w = gw.iloc[i, 32]
        df.loc[df.Rcls == clss, "W_Gen"] = w
        df.loc[df.Rcls == clss, "s_wpls_gen"] = s_w
        df.loc[df.Rcls == clss, "var_wpls_gen"] = v_w
        df.round(4)
    return df


def gen_weights(df: pd.DataFrame, gen_cls: int) -> pd.DataFrame:
    """_summary_
    Args:
        df (pd.DataFrame):
        gen_cls (int): List of generalized class values
    Returns:
        df_gen_wgts (pd.DataFrame): Dataframe with positives weights of associations for generalized classes
    Raises:

    """
    df_rc = df.groupby("Rcls")
    df_rc_ = df_rc.get_group(gen_cls)
    df_gen_wgts = (
        df_rc_.assign(cmltv_dep_cnt=lambda df_rc_: df_rc_.Point_Count.cumsum())
        .assign(cmltv_cnt=lambda df_rc_: df_rc_.Count.cumsum())
        .assign(cmltv_no_dep_cnt=lambda df_rc_: df_rc_.No_Dep_Cnt.cumsum())
        .iloc[[-1]]
    )

    return (
        df_gen_wgts.assign(Num_wpls=lambda df_gen_wgts: df_gen_wgts.cmltv_dep_cnt / df_gen_wgts.Total_Deposits)
        .assign(Deno_wpls=lambda df_gen_wgts: df_gen_wgts.cmltv_no_dep_cnt / df_gen_wgts.Tot_No_Dep_Cnt)
        .assign(W_Gen=lambda df_gen_wgts: np.log(df_gen_wgts.Num_wpls) - np.log(df_gen_wgts.Deno_wpls))
        .assign(var_wpls_gen=lambda df_gen_wgts: (1 / df_gen_wgts.cmltv_dep_cnt) + (1 / df_gen_wgts.cmltv_no_dep_cnt))
        .assign(s_wpls_gen=lambda df_gen_wgts: np.sqrt(df_gen_wgts.var_wpls_gen))
    )


def _weights_arrays(evidential_raster: rasterio.io.DatasetReader, weights_df: pd.DataFrame, col_names: List) -> Tuple[List, dict]:
    """Calls the raster_arrays function to convert the generalized weights dataaframe to numpy arrays.

    Args:
        evidential_raster (rasterio.io.DatasetReader): The evidential raster with spatial resolution and extent identical to that of the deposit_raster.
        weights_df (pd.DataFrame): Dataframe with the weights.
        col_names (List): Columns to generate the arrays from.

    Returns:
        gen_arrys (List): List of individual raster object arrays for generalized or unique classes, generalized weights and standard deviation of generalized weights
        raster_meta (dict): Raster array's metadata.
    """
    raster_meta = evidential_raster.meta.copy()
    list_cols = list(weights_df.columns)
    nan_row = {val: -1.0e09 for val in list_cols}
    nan_row_df = pd.DataFrame.from_dict(nan_row, orient="index")
    nan_row_df_t = nan_row_df.T
    weights_df_nan = pd.concat([nan_row_df_t, weights_df])
    class_rstr, w_gen_rstr, std_rstr = map(functools.partial(_raster_array, evidential_raster, weights_df_nan), col_names)
    gen_arrys = [class_rstr, w_gen_rstr, std_rstr]
    return gen_arrys, raster_meta


def _raster_array(evidential_raster: rasterio.io.DatasetReader, weights_df_nan: pd.DataFrame, col: str) -> np.ndarray:
    """Converts the generalized weights dataframe to numpy arrays with the extent and shape of the input raster

    Args:
        evidential_raster (rasterio.io.DatasetReader): The evidential raster with spatial resolution and extent identical to that of the deposit_raster.
        weights_df_nan (pd.DataFrame): Generalized weights dataframe with info on NaN data also.
        col (str): Columns to use for generation of raster object arrays.

    Returns:
        np.ndarray: Individual raster object arrays for generalized or unique classes, generalized weights and standard deviation of generalized weights
    """
    # raster_meta = evidential_raster.meta.copy()
    raster_array = np.array(evidential_raster.read(1))
    weights_mapping_dict = {}
    weights_mapping_dict = pd.Series(weights_df_nan.loc[:, col], index=weights_df_nan.Class).to_dict()
    replace_array = np.array([list(weights_mapping_dict.keys()), list(weights_mapping_dict.values())])
    raster_array_wgts = raster_array.reshape(-1)
    mask_array = np.isin(raster_array_wgts, replace_array[0, :])
    ss_rplc_array = np.searchsorted(replace_array[0, :], raster_array_wgts[mask_array])
    raster_array_replaced = replace_array[1, ss_rplc_array]
    raster_array_replaced = raster_array_replaced.reshape(raster_array.shape)
    return raster_array_replaced
