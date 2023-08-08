from typing import Tuple
import pandas as pd
from enum import Enum

class WeightsOfEvidenceType(Enum):
    Unique = 0
    CumulativeAscending = 1
    CumulativeDescending = 2

    def __str__(self):
        return f'{self.name.lower()}({self.value})'
    
    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other
    
        if isinstance(other, WeightsOfEvidenceType):
            return self is other
    
        return False

def _weights_type(
    df: pd.DataFrame, nan_val: float, w_type: int = 0
) -> Tuple [pd.DataFrame, pd.DataFrame]:

    df.loc[df.Class <= nan_val, 'N_cls'] = 'NaN' # not needed as such
    df.loc[df.Class > nan_val, 'N_cls'] = 'Data'
    df_rcl=df.groupby('N_cls')
    df_rcl_dt = df_rcl.get_group('Data')
    df_rcl_nan = df_rcl.get_group('NaN')

    pd.set_option('mode.chained_assignment', None)        
    if w_type == WeightsOfEvidenceType.Unique:
        #returns the df with the data classes  for categorical weights calculations
        return df_rcl_dt, df_rcl_nan
        #for sorting of classes for numerical weights calculations
    else: #w_type != 0:
        df_rcl_dt.rename(columns = {"Point_Count":"Act_Point_Count"}, inplace = True)
        if w_type == WeightsOfEvidenceType.CumulativeAscending:
            df_srtd = df_rcl_dt.sort_values(by = "Class", ascending=True)
            #return df_srtd
        elif w_type == WeightsOfEvidenceType.CumulativeDescending:
            df_srtd = df_rcl_dt.sort_values(by = "Class", ascending=False)    
    
        pd.set_option('mode.chained_assignment', None)   
        df_data_srtd = (df_srtd
                .assign(Point_Count = lambda df_srtd: df_srtd.Act_Point_Count.cumsum().replace(0,0.0001))
                .assign(cmltv_cnt = lambda df_srtd: df_srtd.Count.cumsum())
                .assign(No_Dep_Cnt = lambda df_srtd: df_srtd.No_Dep_Cnt.cumsum().replace(0, 0.0001))
                 )
        return df_data_srtd, df_rcl_nan

        
   
def weights_type(
        df: pd.DataFrame,  nan_val: float, w_type: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identifies NoData and separates it out from subsequent calculations. Based on the type of weights selected by the user, the function performs the sorting and cumulative count calculations.
    Args:
        df (pandas.DataFrame): The dataframe with basic calculations performed in the basic_calculations function
        nan_val (float): value of no data
        w_type(int, def = 0): 0 = unique weights, 1 = cumulative ascending weights, 2 = cumulative descending weights
    Returns:
        df_data (pandas.DataFrame): The dataframe with data values and sorted is weights calculations type is numerical (i.e., w_type = 1 or 2)
        df_nan (pandas.DataFrame): The dataframe with information on NoData
            
    """
    w_type_acc = [0,1,2]
    if w_type not in w_type_acc:
        raise ValueError("Accepted values of w_type are 0, 1, 2 for unique, cumulative ascending and cumulative descending weights respectively")
    else:
        df_data, df_nan = _weights_type(df, nan_val, w_type)
        return df_data, df_nan
