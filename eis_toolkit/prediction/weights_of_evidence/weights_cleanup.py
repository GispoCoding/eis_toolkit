import pandas as pd


def _weights_cleanup(
    df: pd.DataFrame,  w_type: int = 0
) -> pd.DataFrame:

    drop_cols = ['No_Dep_Cnt', 'Total_Area', 'Total_Deposits', 'Tot_No_Dep_Cnt',
                 'Dep_outsidefeat', 'Non_feat_Pxls', 'Non_Feat_Non_Dep',
                 'N_cls', 'Num_wpls', 'Deno_wpls', 'var_wpls', 'Num_wmns',
                 'Non_Feat_Pxls_cm', 'Deno_wmns', 'var_wmns', 'var_wpls_gen',
                 'cmltv_dep_cnt', 'cmltv_cnt', 'cmltv_no_dep_cnt'
                 ]

    if w_type != 0:
        cols_rename = {'Class': 'Class', 
                       'Point_Count': 'Cmltv. Point Count', 
                       'Count': 'Cmltv. Count',
                       'Act_Count': 'Count_', 
                       'Act_Point_Count': 'Point Count_',
                       'wpls': 'WPlus', 
                       's_wpls': 'S_WPlus', 
                       'wmns': 'WMinus', 
                       's_wmns': 'S_WMinus',
                       'contrast': 'Contrast', 
                       's_contrast': 'S_Contrast', 
                       'Stud_Cont': 'Stud. Contrast',
                       'Rcls': 'Gen_Class', 
                       'W_Gen': 'Gen_Weights', 
                       's_wpls_gen': 'S_Gen_Weights'
                       }
    else:
        cols_rename = {'Class': 'Class', 
                       'Point_Count': 'Point Count', 
                       'Count': 'Count',
                       'wpls': 'WPlus', 
                       's_wpls': 'S_WPlus', 
                       'wmns': 'WMinus', 
                       's_wmns': 'S_WMinus',
                       'contrast': 'Contrast', 
                       's_contrast': 'S_Contrast', 
                       'Stud_Cont': 'Stud. Contrast',
                       }
    df = (df.drop([col for col in drop_cols if col in df.columns], axis=1)
            .rename(columns=cols_rename)
            .round(4)
            )
    return df


def weights_cleanup(
    df: pd.DataFrame,  w_type: int = 0
) -> pd.DataFrame:
    """ Removes unnecessary columns and creates a clean dataframe with important spatial associations quantities.
        For caterogical data with weights calculations type 'unique', this function is called after the weights calculations.
        For numerical data this function is called after reclassification and generalized weights calculations.

    Args:
        df (pandas.DataFrame): Data frame with all the calculations; obtained from the contrast function (for categorical data) or from the generalized weights function (for ordinal data)
        w_type (int, def = 0): 0 = unique weights, 1 = cumulative ascending weights, 2 = cumulative descending weights
    Returns:
        df_weights (pandas.DataFrame): Final dataframe with only the necessary values.
    
    """

    df_weights = _weights_cleanup(df, w_type)
    return df_weights
