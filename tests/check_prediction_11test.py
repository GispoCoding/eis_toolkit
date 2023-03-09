
import pytest
# import numpy as np
import sys
from pathlib import Path

scripts = r'/eis_toolkit'  #/eis_toolkit/conversions'
sys.path.append (scripts)

import geopandas as gpd
import pandas as pd
from eis_toolkit.conversions.import_featureclass import *
from eis_toolkit.conversions.import_grid import *
from eis_toolkit.transformations.separation import *
from eis_toolkit.transformations.nodata_replace import *
from eis_toolkit.transformations.onehotencoder import *
from eis_toolkit.transformations.unification import *
from eis_toolkit.model_training.sklearn_randomforest_classifier import *
from eis_toolkit.model_training.sklearn_model_fit import *
from eis_toolkit.prediction_methods.sklearn_model_prediction import *
from eis_toolkit.validation.sklearn_model_validations import *
from eis_toolkit.validation.sklearn_model_crossvalidation import *
from eis_toolkit.validation.sklearn_model_importance import *
from eis_toolkit.file.export_files import *
from eis_toolkit.file.import_files import *
from eis_toolkit.checks.sklearn_check_prediction import *
from eis_toolkit.transformations.split import *

from eis_toolkit.exceptions import NonMatchingCrsException, NotApplicableGeometryTypeException, InvalideContentOfInputDataFrame
from eis_toolkit.exceptions import MissingFileOrPath

#################################################################
# import of data from import_featureclass or import_grid
# fc or csv:
parent_dir = Path(__file__).parent
name_fc = str(parent_dir.joinpath(r'data/shps/EIS_gp.gpkg'))
layer_name = r'Occ_2'
name_csv = str(parent_dir.joinpath(r'data/csv/Test_Test.csv'))

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

# columns , df , urdf , metadata = import_featureclass(fields = fields_fc , file = name_fc , layer = layer_name)
columns, df, urdf, metadata = import_featureclass(fields = fields_csv, file = name_csv, decimalpoint_german = True) 
#columns , df , metadata = import_grid(grids = grids) 
# Separation
Xvdf, Xcdf, ydf, igdf = separation(df = df, fields = columns) 
# nodata_replacement of 
Xcdf = nodata_replace(df = Xcdf, rtype = 'most_frequent')
Xvdf = nodata_replace(df = Xvdf, rtype = 'mean')
# onehotencoder
Xdf_enh, eho = onehotencoder(df = Xcdf)
# unification
Xdf = unification(Xvdf = Xvdf, Xcdf = Xdf_enh)
# model
sklearnMl = sklearn_randomforest_classifier(oob_score = True)
# fit
sklearnMl = sklearn_model_fit (sklearnMl = sklearnMl, Xdf = Xdf, ydf = ydf)
# validation
#validation , confusion , comparison , myMl = sklearn_model_validations (sklearnMl = sklearnMl , Xdf = Xdf , ydf = ydf , comparison = True , confusion_matrix = True , test_size = 0.2)
# crossvalidation
#cv = sklearn_model_crossvalidation (sklearnMl = sklearnMl , Xdf = Xdf , ydf = ydf)
# importance
#importance = sklearn_model_importance (sklearnMl = sklearnMl , Xdf = Xdf , ydf = ydf)

# export files
parent_dir = Path(__file__).parent
path = str(parent_dir.joinpath(r'data'))
filesdict = export_files(
    name = 'test_csv',
    path = path,
    sklearnMl = sklearnMl,
    sklearnOhe = eho,
    myFields = columns,
    decimalpoint_german = True,
    new_version = False,
)

# import files
sklearnMlp, sklearnOhep, myFieldsp, kerasMlp, kerasOhep = import_files(
    sklearnMl_file = filesdict['sklearnMl'],
    sklearnOhe_file = filesdict['sklearnOhe'],
    myFields_file = filesdict['myFields'], 
    )

# split
Xdf_train, Xdf_test, ydf_train, ydf_test = split(Xdf = df, test_size = 10)

Xvdf_test, Xcdf_test, ydf_dmp, igdf_test = separation(df = Xdf_test, fields = myFieldsp) 
# nodata_replacement of 
Xcdf_test = nodata_replace(df = Xcdf_test, rtype = 'most_frequent')
Xvdf_test = nodata_replace(df = Xvdf_test, rtype = 'mean') 
# onehotencoder
Xdf_enht, eho = onehotencoder(df = Xcdf_test, ohe = sklearnOhep)
# unification
Xdf_tst = unification(Xvdf = Xvdf_test, Xcdf = Xdf_enht)

#################################################################

def test_sklearn_check_prediction():
    """Test functionality of save import files."""

    Xdf = sklearn_check_prediction(sklearnMl= sklearnMlp, Xdf = Xdf_tst)
    assert (isinstance(Xdf,pd.DataFrame))

    # new order
    Xdf_sort = Xdf_tst.sort_index(axis=1)
    Xdf = sklearn_check_prediction(sklearnMl= sklearnMlp, Xdf = Xdf_sort)
    assert (isinstance(Xdf,pd.DataFrame))

def test_sklearn_check_prediction_error():
    """Test wrong arguments."""

    Xdf_wrong = Xdf_test.drop(columns=['K_H'])
    with pytest.raises(InvalideContentOfInputDataFrame):
        X = sklearn_check_prediction(sklearnMl= sklearnMlp, Xdf = Xdf_wrong)
    
    with pytest.raises(InvalidParameterValueException):
        X = sklearn_check_prediction(sklearnMl= Xdf_tst, Xdf = Xdf_tst)
    with pytest.raises(InvalidParameterValueException):
        X = sklearn_check_prediction(sklearnMl= sklearnMlp, Xdf = 999)

test_sklearn_check_prediction()
test_sklearn_check_prediction_error()
