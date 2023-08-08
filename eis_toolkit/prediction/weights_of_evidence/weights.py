import pandas as pd
import numpy as np


def positive_weights(
    df: pd.DataFrame
) -> pd.DataFrame:
    """ Calculates positive weights of spatial associations between the input data and the points
    Args:
        df (pandas.DataFrame): The dataframe containing data values, obtained from the weights_type function
    Returns:
         df_wpls (pandas.DataFrame): The dataframe with positive weights of spatial associaton for each data class
    Raises:

    """
    pd.set_option('mode.chained_assignment', None)
    df_wpls = (df
               .assign(Point_Count=lambda df: df.Point_Count.
                       replace(0, 0.0001))
               .assign(Num_wpls=lambda df: (df.Point_Count/df.Total_Deposits)
                       .replace(0, 0.0001).replace(1, 1.0001))
               .assign(Deno_wpls=lambda df: df.No_Dep_Cnt/df.Tot_No_Dep_Cnt)
               .assign(wpls=lambda df: np.log(df.Num_wpls)-np.log(df.Deno_wpls))
               .assign(var_wpls=lambda df: (1/df.Point_Count)+(1/df.No_Dep_Cnt))
               .assign(s_wpls=lambda df: np.sqrt(df.var_wpls))
               )
    return df_wpls


def negative_weights(
    df_: pd.DataFrame, w_type: int = 0
) -> pd.DataFrame:
    """ Calculates negative weights of spatial associations between the input data and the points.
    Args: 
        df (pandas.DataFrame): The dataframe containing the positive weights for the data values; obtained from the positive_weights function
        w_type (int = 0):  0 = unique weights, 1 = cumulative ascending weights, 2 = cumulative descending weights.
    Returns:
        df_wmns (pandas.DataFrame): The dataframe with the negative weights of spatial association for each class.
    Raises:

    """
    df = df_.copy()
    pd.set_option('mode.chained_assignment', None)
    if w_type != 0:
        df.rename(columns={"Count": "Act_Count",
                  "cmltv_cnt": "Count"}, inplace=True)

    df_wmns = (df
               .assign(Num_wmns=lambda df: (df.Total_Deposits - (df.Point_Count/df.Total_Deposits))
                       .replace(0, 0.0001))
               .assign(Dep_outsidefeat=lambda df: (df.Total_Deposits - df.Point_Count)
                       .replace(0, 0.0001))
               .assign(Non_Feat_Pxls_cm=lambda df: (df.Total_Area - df.Count)
                       .replace(0, 0.0001))
               .assign(Non_Feat_Non_Dep=lambda df: (df.Non_Feat_Pxls_cm - df.Dep_outsidefeat)
                       .replace(0, 0.0001))
               .assign(Deno_wmns=lambda df: (df.Non_Feat_Pxls_cm - (df.Dep_outsidefeat/df.Tot_No_Dep_Cnt))
                       .replace(0, 0.0001))
               .assign(wmns=lambda df: np.log(df.Num_wmns)-np.log(df.Deno_wmns))
               .assign(var_wmns=lambda df: (1/df.Dep_outsidefeat)+(1/df.Non_Feat_Non_Dep))
               .assign(s_wmns=lambda df: np.sqrt(df.var_wmns))
               .round(4)
               )  # some of these calculations could be clubbed together
    # this should be before 'wmns' assignment, check at some point
    df_wmns.loc[df_wmns["Num_wmns"] ==
                df_wmns["Deno_wmns"], "Num_wmns"] = 1.0001
    return df_wmns


def contrast(df: pd.DataFrame
             ) -> pd.DataFrame:
    """Calculates the contrast and the studentized contrast values from the postive and negative spatial associations quantified as weights in the positive_weights and negative_weights functions
    Args:
        df (pandas.DataFrame): The dataframe containing the positive and negative weights of spatial associations; obtained from the negative_weights function 
    Returns:
        df_cont (pandas.DataFrame): The dataframe with contrast and studentized contrast values  
    """

    df_cont = (df
               .assign(contrast=lambda df: df.wpls - df.wmns)
               .assign(s_contrast=lambda df: np.sqrt(df.var_wpls + df.var_wmns))
               .assign(Stud_Cont=lambda df: df.contrast/df.s_contrast)
               .round(3)
               )
    #df_stud_cont = df_cont[['Class', 'contrast', 's_contrast', 'Stud_Cont']]
    return df_cont
