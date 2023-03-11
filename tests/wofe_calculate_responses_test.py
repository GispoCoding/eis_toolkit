import pytest
import numpy as np
from pathlib import Path
import rasterio
import pandas as pd
from pandas.testing import assert_frame_equal

from eis_toolkit.prediction.weights_of_evidence.calculate_responses import calculate_responses

parent_dir = Path(__file__).parent
print(parent_dir)

# Paths of files to be used as inputs to the calculate responses function
wgts_rst_un_path = parent_dir.joinpath("data/remote/wofe/Merged_Unique.tif")
wgts_rst_asc_path = parent_dir.joinpath("data/remote/wofe/Merged_Ascending_.tif")
wgts_rst_dsc_path = parent_dir.joinpath("data/remote/wofe/Merged_Descending_.tif")
dep_rst_path = parent_dir.joinpath("data/remote/wofe/wofe_dep_nan_.tif")
    
# Path to the file to compare the results of the calculates responses function
merged_pprb_path =  parent_dir.joinpath("data/remote/wofe/Merged_pprbs.tif")

# Actual testing function
def test_calculate_responses():
    """Tests if calculate_responses function works as intended"""
    # expected arrays
    pprb_rstrs = rasterio.open(merged_pprb_path)
    pprb, std, conf = np.array(pprb_rstrs.read(1)), np.array(pprb_rstrs.read(2)), np.array(pprb_rstrs.read(3))
    exp_pprbs = [pprb, std, conf]
    
    #input to calculate_responses function
    test_dep = rasterio.open(dep_rst_path)
    wgts_rstr_paths = [wgts_rst_un_path, wgts_rst_asc_path, wgts_rst_dsc_path]
    arrys_list_from_wgts = []
    for path_rst in wgts_rstr_paths:
        wgts_rst_o = rasterio.open(path_rst)
        pprb_, std_, conf_ = np.array(wgts_rst_o.read(1)), np.array(wgts_rst_o.read(2)), np.array(wgts_rst_o.read(3))
        list_one_block = [pprb_, std_, conf_]
        arrys_list_from_wgts.append(list_one_block)

    # Call the calculate_responses function
    t_pprb_array, t_pprb_std, t_pprb_conf, array_meta = calculate_responses(test_dep, arrys_list_from_wgts)
    result_pprbs = [t_pprb_array, t_pprb_std, t_pprb_conf]

    #compare the results
    for res, exp in zip(result_pprbs, exp_pprbs):
        assert res.all() == exp.all()