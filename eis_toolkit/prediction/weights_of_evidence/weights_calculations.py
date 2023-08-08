from typing import Tuple, List
import rasterio
import pandas as pd
from enum import Enum

from eis_toolkit.prediction.weights_of_evidence.generalized_weights import weights_generalization
from eis_toolkit.prediction.weights_of_evidence.weights_arrays import weights_arrays
from eis_toolkit.prediction.weights_of_evidence.weights_cleanup import weights_cleanup
from eis_toolkit.prediction.weights_of_evidence.weights import positive_weights, negative_weights, contrast
from eis_toolkit.prediction.weights_of_evidence.weights_type import weights_type

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

def _weights_calculations(
        ev_rst: rasterio.io.DatasetReader,
        bsc_clc: pd.DataFrame,
        nan_val: float,
        w_type: int = 0, stud_cont: float = 2
) -> Tuple[pd.DataFrame, List, dict]:

    df_wgts_test, df_nan = weights_type(
        bsc_clc, nan_val, w_type)  # df_nan is not needed
    wpls_df = positive_weights(df_wgts_test)
    wmns_df = negative_weights(wpls_df, w_type)
    contrast_df = contrast(wmns_df)

    if w_type == WeightsOfEvidenceType.Unique:
        cat_wgts = weights_cleanup(contrast_df)
        col_names = ['Class', 'WPlus', 'S_WPlus']
        gen_arrys, rstr_meta = weights_arrays(ev_rst, cat_wgts, col_names)
        return cat_wgts, gen_arrys, rstr_meta
    else:
        num_weights = weights_generalization(contrast_df, w_type, stud_cont,)
        col_names = ['Gen_Class', 'Gen_Weights', 'S_Gen_Weights']
        gen_arrys, rstr_meta = weights_arrays(ev_rst, num_weights, col_names)
        return num_weights, gen_arrys, rstr_meta


def weights_calculations(
        ev_rst: rasterio.io.DatasetReader,
        bsc_clc: pd.DataFrame,
        nan_val:float,
        w_type: int = 0, stud_cont: float = 2
) -> Tuple[pd.DataFrame, List, dict]:
    """ Calculates weights of spatial associations.

    Args:
        ev_rst (rasterio.io.DatasetReader): The evidential raster.
        bsc_clc(pd.DataFrame): Dataframe obtained from basic_calculations function.
        nan_val (float): value of no data
        w_type (int, optional): Accepted values are 0 for unique weights, 1 for cumulative ascending weights, 2 for cumulative descending weights. Defaults to 0.
        stud_cont (float, optional): studentized contrast value to be used for genralization of classes. Not needed if w_type = 0. Defaults to 2.

    Returns:
        weights_df (pd.DataFrame): Dataframe with weights of spatial association between the input rasters.
        gen_arrays (List): List of output raster arrays with generalized or unique classes, generalized weights and standard deviation of generalized weights.
        raster_meta (dict): Raster array's metadata.

    """

    weights_df, gen_arrys, raster_meta = _weights_calculations(
        ev_rst, bsc_clc, nan_val, w_type, stud_cont)
    return weights_df, gen_arrys, raster_meta
