from typing import Tuple, List
import rasterio
import pandas as pd

from eis_toolkit.prediction.weights_of_evidence.calculate_weights import calculate_weights
from eis_toolkit.prediction.weights_of_evidence.basic_calculations import basic_calculations
from eis_toolkit.checks.crs import check_matching_crs
from eis_toolkit.exceptions import NonMatchingCrsException

def _weights_calculations(
        ev_rst: rasterio.io.DatasetReader,
        dep_rst: rasterio.io.DatasetReader,
        nan_val: float,
        w_type: int = 0,
        stud_cont: float = 2
) -> Tuple[pd.DataFrame, List, dict]:

    bsc_clc_df = basic_calculations(ev_rst, dep_rst, nan_val)
    weights_df, raster_gen, raster_meta = calculate_weights(
        ev_rst, bsc_clc_df, nan_val, w_type, stud_cont)
    return weights_df, raster_gen, raster_meta


def weights_calculations(
    ev_rst: rasterio.io.DatasetReader,
    dep_rst: rasterio.io.DatasetReader,
    nan_val: float,
    w_type: int = 0,
    stud_cont: float = 2
) -> Tuple[pd.DataFrame, List, dict]:
    """Calculates weights of spatial associations.

    Args:
        ev_rst (rasterio.io.DatasetReader): The evidential raster with spatial resolution and extent identical to that of the dep_rst.
        dep_rst (rasterio.io.DatasetReader): Raster representing the mineral deposits or occurences point data. 
        nan_val (float): value of no data
        w_type (int, optional): Accepted values are 0 for unique weights, 1 for cumulative ascending weights, 2 for cumulative descending weights. Defaults to 0.
        stud_cont (float, optional): studentized contrast value to be used for genralization of classes. Not needed if w_type = 0. Defaults to 2.

    Returns:
        weights_df (pd.DataFrame): Dataframe with weights of spatial association between the input rasters
        raster_gen (List): List of output raster arrays with generalized or unique classes, generalized weights and standard deviation of generalized weights
        raster_meta (dict): Raster array's metadata.

    Raises:         
        ValueError: Accepted values of w_type are 0, 1, 2 for unique, cumulative ascending and cumulative descending weights respectively.
        The below exceptions will be incorporated into the function later as the development for other related functions progresses in the toolkit.
        NonMatchingCrsException: The input rasters are not in the same crs
        InvalidParameterValueException: Accepted values of w_type are 0, 1, 2 for unique, cumulative ascending and cumulative descending weights respectively. (status - pending)
        NonMatchingTransformException: The input rasters do not have the same cell size and/or same extent (status - pending)
        NonMatchingCoRegistrationException: The input rasters are not coregistered (status - pending)
    """

    w_type_acc = [0, 1, 2]
    if w_type not in w_type_acc:
        raise ValueError(
            "Invalid parameter values. Accepted values of w_type are 0, 1, 2 for unique, cumulative ascending and cumulative descending weights respectively")
    """
    if not check_matching_crs(
            objects=[ev_rst, dep_rst]
    ):
        raise NonMatchingCrsException
    """
    weights_df, raster_gen, raster_meta = _weights_calculations(ev_rst, dep_rst, nan_val, w_type, stud_cont)

    return weights_df, raster_gen, raster_meta

