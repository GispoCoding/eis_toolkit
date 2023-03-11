from typing import Tuple
import pandas as pd

def _weights_type(
    df: pd.DataFrame, w_type: int = 0
) -> Tuple [pd.DataFrame, pd.DataFrame]:

    df.loc[df.Class <= -1000000000.0, 'N_cls'] = 'NaN' # not needed as such
    df.loc[df.Class > -1000000000.0, 'N_cls'] = 'Data'
    df_rcl=df.groupby('N_cls')
    df_rcl_dt = df_rcl.get_group('Data')
    df_rcl_nan = df_rcl.get_group('NaN')

    pd.set_option('mode.chained_assignment', None)        
    if (w_type == 0):
        #returns the df with the data classes  for categorical weights calculations
        return df_rcl_dt, df_rcl_nan
        #for sorting of classes for numerical weights calculations
    elif (w_type != 0):
        df_rcl_dt.rename(columns = {"Point_Count":"Act_Point_Count"}, inplace = True)
        if (w_type == 1):
            df_srtd = df_rcl_dt.sort_values(by = "Class", ascending=True)
            #return df_srtd
        elif (w_type == 2):
            df_srtd = df_rcl_dt.sort_values(by = "Class", ascending=False)    
    
        pd.set_option('mode.chained_assignment', None)   
        return (df_srtd
                .assign(Point_Count = lambda df_srtd: df_srtd.Act_Point_Count.cumsum().replace(0,0.0001))
                .assign(cmltv_cnt = lambda df_srtd: df_srtd.Count.cumsum())
                .assign(No_Dep_Cnt = lambda df_srtd: df_srtd.No_Dep_Cnt.cumsum().replace(0, 0.0001)),
                df_rcl_nan
                )
   
def weights_type(
        df: pd.DataFrame,  w_type: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identifies NoData and separates it out from subsequent calculations. Based on the type of weights selected by the user, the function performs the sorting and cumulative count calculations.
    Args:
        df (pandas.DataFrame): The dataframe with basic calculations performed in the basic_calculations function
        w_type(int, def = 0): 0 = unique weights, 1 = cumulative ascending weights, 2 = cumulative descending weights
    Returns:
        df_data (pandas.DataFrame): The dataframe with data values and sorted is weights calculations type is numerical (i.e., w_type = 1 or 2)
        df_nan (pandas.DataFrame): The dataframe with information on NoData
    Raises:
        
    """
    w_type_acc = [0,1,2]
    if w_type not in w_type_acc:
        raise ValueError("Accepted values of w_type are 0, 1, 2 for unique, cumulative ascending and cumulative descending weights respectively")
    else:
        df_data, df_nan = _weights_type(df, w_type)
        return df_data, df_nan
