
# all_separation_test.py
##############################
import pytest
# import numpy as np
import sys
from pathlib import Path

scripts = r'/eis_toolkit'  #/eis_toolkit/conversions'
sys.path.append (scripts)

import geopandas as gpd
import pandas as pd
from eis_toolkit.conversions.all_import_featureclass import *
from eis_toolkit.conversions.all_import_grid import *
from eis_toolkit.transformations.all_separation import *
#from eis_toolkit.exceptions import NonMatchingCrsException, NotApplicableGeometryTypeException

#################################################################
# import of data from all_import_featureclass or all_import_grid
# fc or csv:
parent_dir = Path(__file__).parent
name_fc = str(parent_dir.joinpath(r'data/shps/EIS_gp.gpkg'))
layer_name = r'Occ_2'
name_csv = str(parent_dir.joinpath(r'data/csv/Trainings_Test.csv'))

# grid:
parent_dir = Path(__file__).parent
name_K = str(parent_dir.joinpath(r'data/Primary_data/Rad/IOCG_Gm_Rd_K_.tif'))
name_Th = str(parent_dir.joinpath(r'data/Primary_data/Rad/IOCG_Gm_Rd_Th_eq_.tif'))
name_U = str(parent_dir.joinpath(r'data/Primary_data/Rad/IOCG_Gm_Rd_U_eq_.tif'))
name_target = str(parent_dir.joinpath(r'data/Primary_data/Rad/IOCG_Gm_Rd_Total_Count_.tif'))

#grids and grid-types for X (training based on tif-files)
grids =  [{'name':'Total','type':'t','file':name_target},
 {'name':'Kalium', 'file':name_K, 'type':'v'},
 {'name':'Thorium', 'file':name_Th, 'type':'v'},
 {'name':'Uran', 'file':name_U, 'type':'v'}]

#columns and column-types for X (training based on a geopackage-layer)
fields_fc=  {'OBJECTID':'i', 'ID':'n', 'Name':'n', 'Alternativ':'n', 'Easting_EU':'n', 'Northing_E':'n',
       'Easting_YK':'n', 'Northing_Y':'n', 'Commodity':'c', 'Size_by_co':'c', 'All_main_c':'n',
       'Other_comm':'n', 'Ocurrence_':'n', 'Mine_statu':'n', 'Discovery_':'n', 'Commodity_':'n',
       'Resources_':'n', 'Reserves_t':'n', 'Calc_metho':'n', 'Calc_year':'n', 'Current_ho':'n',
       'Province':'n', 'District':'n', 'Deposit_ty':'n', 'Host_rocks':'n', 'Wall_rocks':'n',
       'Deposit_sh':'c', 'Dip_azim':'v', 'Dip':'t', 'Plunge_azi':'v', 'Plunge_dip':'v', 'Length_m':'v',
       'Width_m':'n', 'Thickness_':'v', 'Depth_m':'n', 'Date_updat':'n', 'geometry':'g'}

#columns and column-types for X (training based on a csv file)
fields_csv=  {'LfdNr':'i','Tgb':'t','TgbNr':'n','SchneiderThiele':'c','SuTNr':'c','Asche_trocken':'v','C_trocken':'n','H_trocken':'v',
       'S_trocken':'v','N_trocken':'v','AS_Natrium_ppm':'v','AS_Kalium_ppm':'v','AS_Calcium_ppm':'v',
       'AS_Magnesium_ppm':'v','AS_Aluminium_ppm':'v','AS_Silicium_ppm':'v','AS_Eisen_ppm':'v','AS_Titan_ppm':'v','AS_Schwefel_ppm':'n',
       'H_S':'v','Na_H':'v','K_H':'v','Ca_H':'v','Mg_H':'v','Al_H':'v','Si_H':'v','Fe_H':'v','Ti_H':'v',
       'Na_S':'v','K_S':'v','Ca_S':'v','Mg_S':'v','AL_S':'v','Si_S':'v','Fe_S':'v',
       'Ti_S':'v','Na_K':'v','Na_Ca':'v','Na_Mg':'v','Na_Al':'v','Si_Na':'v','Na_Fe':'v','Na_Ti':'v','K_Ca':'v','K_Mg':'v',
       'K_Al':'v','Si_K':'v','K_Fe':'v','K_Ti':'v','Ca_Mg':'v','Ca_Al':'v',
       'Si_Ca':'v','Ca_Fe':'v','Ca_Ti':'v','Mg_Al':'v','Si_Mg':'v','Mg_Fe':'v','Mg_Ti':'v','Si_Al':'v',
       'Al_Fe':'v','Al_Ti':'v','Si_Fe':'v','Si_Ti':'v','Fe_Ti':'v'}

# columns , df , urdf , metadata = all_import_featureclass(fields = fields_fc , file = name_fc , layer = layer_name)
columns , df , urdf , metadata = all_import_featureclass(fields = fields_csv , file = name_csv , decimalpoint_german = True) 
#columns , df , metadata = all_import_grid(grids = grids) 

#################################################################

def test_all_separation():
    """Test functionality of separation of imported X (Dataframe)."""
    Xvdf , Xcdf , ydf , igdf = all_separation(df = df, fields = columns) 

    assert ((isinstance(Xcdf,pd.DataFrame)) or (Xcdf is None))
    assert ((isinstance(Xvdf,pd.DataFrame)) or (Xvdf is None))
    assert ((isinstance(ydf,pd.DataFrame)) or (ydf is None))
    assert ((isinstance(igdf,pd.DataFrame)) or (igdf is None))
    if (Xvdf is not None) and (ydf is not None):
        assert len(Xcdf.index) == len(ydf.index) 

def test_all_separation_error():
    """Test wrong arguments of separation of imported X (Dataframe) with wrong arguments."""
    with pytest.raises(InvalidParameterValueException):
        Xvdf , Xcdf , ydf , igdf = all_separation(df = df, fields = 'eins:3') 

test_all_separation()
test_all_separation_error()
