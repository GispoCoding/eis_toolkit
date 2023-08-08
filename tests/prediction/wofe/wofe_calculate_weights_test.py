import pytest
import numpy as np
from pathlib import Path
import rasterio
import pandas as pd
from pandas.testing import assert_frame_equal

from eis_toolkit.prediction.weights_of_evidence.calculate_weights import calculate_weights

parent_dir = Path(__file__).parent.parent.parent
print(parent_dir)

# Paths of files to be used as inputs to the weights_calculations function
ev_rst_path = parent_dir.joinpath("data/remote/wofe/wofe_ev_nan.tif")
dep_rst_path = parent_dir.joinpath("data/remote/wofe/wofe_dep_nan_.tif")

# Paths to the files to compare the results of the weigths_calculations function to
wgts_rst_un_path = parent_dir.joinpath("data/remote/wofe/Merged_Unique.tif")
wgts_rst_asc_path = parent_dir.joinpath("data/remote/wofe/Merged_Ascending_.tif")
wgts_rst_dsc_path = parent_dir.joinpath("data/remote/wofe/Merged_Descending_.tif")

wgts_un_path = parent_dir.joinpath("data/remote/wofe/exp_wgts_un.csv")
wgts_asc_path = parent_dir.joinpath("data/remote/wofe/exp_wgts_asc.csv")
wgts_dsc_path = parent_dir.joinpath("data/remote/wofe/exp_wgts_dsc.csv")

@pytest.mark.parametrize("wgts_type, wgts_df_path, expected",  [(0, wgts_un_path, wgts_rst_un_path), (1, wgts_asc_path, wgts_rst_asc_path), (2, wgts_dsc_path, wgts_rst_dsc_path)])
                     
def test_calculate_weights( wgts_type, wgts_df_path, expected):
    """Tests if calculate_weights function runs as intended; tests for all three weights type"""
    #expected results 
    # weights dataframe
    wgts_ = pd.read_csv(wgts_df_path).set_index('Clss')
    exp_wgts_df = pd.DataFrame(wgts_)
    # weights arrays
    wgts_rstr_ = rasterio.open(expected)
    clss, wgts, std = np.array(wgts_rstr_.read(1)), np.array(wgts_rstr_.read(2)), np.array(wgts_rstr_.read(3))
    exp_list_arr = [clss, wgts, std]
    #inputs to function
    ev_rst_ = rasterio.open(ev_rst_path)
    dep_rst_ = rasterio.open(dep_rst_path)
    #Calling the function
    wgts_df, wgts_arr, rst_meta = calculate_weights(ev_rst=ev_rst_, dep_rst= dep_rst_, nan_val = -1000000000.0, w_type=wgts_type, stud_cont=2)
    assert_frame_equal(wgts_df, exp_wgts_df, check_dtype=False, check_index_type=False)
    for res, exp in zip(wgts_arr, exp_list_arr):
        assert res.all() == exp.all()
    