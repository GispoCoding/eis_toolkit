
# all_import_grid_test.py
####################################
import pytest
import sys
from pathlib import Path

scripts = r'/eis_toolkit'  #/eis_toolkit/conversions'
sys.path.append (scripts)

# import rasterio
import geopandas as gpd
import pandas as pd
from eis_toolkit.conversions.all_import_grid import *
#from eis_toolkit.exceptions import NonMatchingCrsException, NotApplicableGeometryTypeException

# input from GUI:
parent_dir = Path(__file__).parent
name_K = str(parent_dir.joinpath(r'data/Primary_data/Rad/IOCG_Gm_Rd_K_.tif'))
name_Th = str(parent_dir.joinpath(r'data/Primary_data/Rad/IOCG_Gm_Rd_Th_eq_.tif'))
name_U = str(parent_dir.joinpath(r'data/Primary_data/Rad/IOCG_Gm_Rd_U_eq_.tif'))
name_target = str(parent_dir.joinpath(r'data/Primary_data/Rad/IOCG_Gm_Rd_Total_Count_.tif'))
name_wrong = str(parent_dir.joinpath(r'data/Primary_data/Rad/IOCG_Gm_Rd_wrong.tif')) 

#grids and grid-types for X (training based on tif-files)
grids =  [{'name':'Total','type':'t','file':name_target},
 {'name':'Kalium', 'file':name_K, 'type':'v'},
 {'name':'Thorium', 'file':name_Th, 'type':'v'},
 {'name':'Uran', 'file':name_U, 'type':'v'}]

gridwrong = [{'name':'Total','type':'t','file':name_target},
 {'name':'Kalium', 'file':name_K, 'type':'v'},
 {'name':'Thorium', 'file':name_Th, 'type':'v'},
 {'name':'Uran', 'file':name_wrong, 'type':'v'}]

def test_all_import_grid_ok():
    """Test functionality: import of tif files and creating of X as DataFrame"""
    columns, df, metadata= all_import_grid(grids = grids) 

    assert isinstance(columns,dict)
    assert isinstance(df,pd.DataFrame)
    assert len(columns) > 0
    assert len(df.index) > 0 
    assert len(df.columns) > 0
    assert isinstance(metadata,dict)
    assert metadata['height'] > 0

def test_all_import_grid_wrong():
    """Test functionality with wrong filenames."""
    with pytest.raises(InvalidParameterValueException):
        columns , df , metadata = all_import_grid(grids = gridwrong) 

test_all_import_grid_ok()
test_all_import_grid_wrong()


