from pathlib import Path

import pytest
# from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation

# from eis_toolkit.checks.sklearn_check_prediction import sklearn_check_prediction
# from eis_toolkit.conversions.export_featureclass import export_featureclass
# from eis_toolkit.conversions.export_grid import export_grid
from eis_toolkit.conversions.import_featureclass import import_featureclass
# from eis_toolkit.conversions.import_grid import import_grid
from eis_toolkit.exceptions import InvalidParameterValueException  # FileWriteError, FileReadError)
# from eis_toolkit.prediction.sklearn_model_predict_proba import sklearn_model_predict_proba
# from eis_toolkit.prediction.sklearn_randomforest_classifier import sklearn_randomforest_classifier
from eis_toolkit.prediction.sklearn_randomforest_regressor import sklearn_randomforest_regressor
from eis_toolkit.transformations.nodata_replace import nodata_replace
from eis_toolkit.transformations.onehotencoder import onehotencoder
from eis_toolkit.transformations.separation import separation
# from eis_toolkit.transformations.split import split
from eis_toolkit.transformations.unification import unification
from eis_toolkit.validation.sklearn_model_crossvalidation import sklearn_model_crossvalidation

# scripts = r"/eis_toolkit"
# sys.path.append(scripts)


# from eis_toolkit.file.export_files import export_files
# from eis_toolkit.file.import_files import import_files
# from eis_toolkit.prediction.sklearn_model_fit import sklearn_model_fit
# from eis_toolkit.prediction.sklearn_model_prediction import sklearn_model_prediction


# from eis_toolkit.validation.sklearn_model_importance import sklearn_model_importance
# from eis_toolkit.validation.sklearn_model_validations import sklearn_model_validations
# from eis_toolkit.transformations.nodata_remove import nodata_remove


#################################################################
# import of data from import_featureclass or import_grid
# fc or csv:
parent_dir = Path(__file__).parent.parent
name_fc = str(parent_dir.joinpath(r"data/shps/EIS_gp.gpkg"))
layer_name = r"Occ_2"
name_csv = str(parent_dir.joinpath(r"data/csv/Test_Test.csv"))

# grid:
# parent_dir = Path(__file__).parent.parent
# name_K = str(parent_dir.joinpath(r"data/Primary_data/Rad/IOCG_Gm_Rd_K_.tif"))
# name_Th = str(parent_dir.joinpath(r"data/Primary_data/Rad/IOCG_Gm_Rd_Th_eq_.tif"))
# name_U = str(parent_dir.joinpath(r"data/Primary_data/Rad/IOCG_Gm_Rd_U_eq_.tif"))
# name_target = str(parent_dir.joinpath(r"data/Primary_data/Rad/IOCG_Gm_Rd_Total_Count_.tif"))

# # grids and grid-types for X (training based on tif-files)
# grids = [
#     {"name": "Total", "type": "t", "file": name_target},
#     {"name": "Kalium", "file": name_K, "type": "v"},
#     {"name": "Thorium", "file": name_Th, "type": "v"},
#     {"name": "Uran", "file": name_U, "type": "v"},
# ]

name_tif1 = str(parent_dir.joinpath(r"data/test1.tif"))
name_tif2 = str(parent_dir.joinpath(r"data/test2.tif"))
name_tif3 = parent_dir.joinpath(r"data/test1.tif")

grids = [
    {"name": "targe", "type": "t", "file": name_tif1},
    {"name": "test1", "file": name_tif2, "type": "v"},
    {"name": "test2", "file": name_tif3, "type": "v"},
]

# columns and column-types for X (training based on a geopackage-layer)
fields_fc = {
    "OBJECTID": "i",
    "ID": "n",
    "Name": "n",
    "Alternativ": "n",
    "Easting_EU": "n",
    "Northing_E": "n",
    "Easting_YK": "n",
    "Northing_Y": "n",
    "Commodity": "c",
    "Size_by_co": "c",
    "All_main_c": "n",
    "Other_comm": "n",
    "Ocurrence_": "n",
    "Mine_statu": "n",
    "Discovery_": "n",
    "Commodity_": "n",
    "Resources_": "n",
    "Reserves_t": "n",
    "Calc_metho": "n",
    "Calc_year": "n",
    "Current_ho": "n",
    "Province": "n",
    "District": "n",
    "Deposit_ty": "n",
    "Host_rocks": "n",
    "Wall_rocks": "n",
    "Deposit_sh": "c",
    "Dip_azim": "v",
    "Dip": "t",
    "Plunge_azi": "v",
    "Plunge_dip": "v",
    "Length_m": "v",
    "Width_m": "n",
    "Thickness_": "v",
    "Depth_m": "n",
    "Date_updat": "n",
    "geometry": "g",
}

# columns and column-types for X (training based on a csv file)
fields_csv = {
    "LfdNr": "i",
    "Tgb": "t",
    "TgbNr": "n",
    "SchneiderThiele": "c",
    "SuTNr": "c",
    "Asche_trocken": "v",
    "C_trocken": "n",
    "H_trocken": "v",
    "S_trocken": "v",
    "N_trocken": "v",
    "AS_Natrium_ppm": "v",
    "AS_Kalium_ppm": "v",
    "AS_Calcium_ppm": "v",
    "AS_Magnesium_ppm": "v",
    "AS_Aluminium_ppm": "v",
    "AS_Silicium_ppm": "v",
    "AS_Eisen_ppm": "v",
    "AS_Titan_ppm": "v",
    "AS_Schwefel_ppm": "n",
    "H_S": "v",
    "Na_H": "v",
    "K_H": "v",
    "Ca_H": "v",
    "Mg_H": "v",
    "Al_H": "v",
    "Si_H": "v",
    "Fe_H": "v",
    "Ti_H": "v",
    "Na_S": "v",
    "K_S": "v",
    "Ca_S": "v",
    "Mg_S": "v",
    "AL_S": "v",
    "Si_S": "v",
    "Fe_S": "v",
    "Ti_S": "v",
    "Na_K": "v",
    "Na_Ca": "v",
    "Na_Mg": "v",
    "Na_Al": "v",
    "Si_Na": "v",
    "Na_Fe": "v",
    "Na_Ti": "v",
    "K_Ca": "v",
    "K_Mg": "v",
    "K_Al": "v",
    "Si_K": "v",
    "K_Fe": "v",
    "K_Ti": "v",
    "Ca_Mg": "v",
    "Ca_Al": "v",
    "Si_Ca": "v",
    "Ca_Fe": "v",
    "Ca_Ti": "v",
    "Mg_Al": "v",
    "Si_Mg": "v",
    "Mg_Fe": "v",
    "Mg_Ti": "v",
    "Si_Al": "v",
    "Al_Fe": "v",
    "Al_Ti": "v",
    "Si_Fe": "v",
    "Si_Ti": "v",
    "Fe_Ti": "v",
}

columns, df, urdf, metadata = import_featureclass(fields=fields_fc, file=name_fc, layer=layer_name)
# columns, df, urdf, metadata = import_featureclass(fields = fields_csv, file = name_csv, decimalpoint_german = True)
# columns , df , metadata = import_grid(grids = grids)
# nodata_:remove
# df, nodatmask = nodata_remove(df = df,)            # for images only
# Separation
Xvdf, Xcdf, ydf, igdf = separation(df=df, fields=columns)
# nodata_replacement of
Xcdf = nodata_replace(df=Xcdf, rtype="most_frequent")
Xvdf = nodata_replace(df=Xvdf, rtype="mean")
ydf = nodata_replace(df=ydf, rtype="most_frequent")
# onehotencoder
Xdf_enh, eho = onehotencoder(df=Xcdf)
# unification
Xdf = unification(Xvdf=Xvdf, Xcdf=Xdf_enh)
# model
# sklearnMl = sklearn_randomforest_classifier(oob_score = True)
sklearnMl = sklearn_randomforest_regressor(oob_score=True)

#################################################################


def test_sklearn_model_crossvalidation():
    """Test functionality of crosvalidation."""

    cv = sklearn_model_crossvalidation(sklearnMl=sklearnMl, Xdf=Xdf, ydf=ydf)

    assert isinstance(cv, dict)


def test_sklearn_model_crossvalidation_error():
    """Test wrong arguments."""

    with pytest.raises(InvalidParameterValueException):
        sklearn_model_crossvalidation(sklearnMl=Xdf_enh, Xdf=Xdf, ydf=ydf, cv=4)
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearn_model_crossvalidation(sklearnMl=sklearnMl, Xdf="_", ydf=ydf)
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearn_model_crossvalidation(sklearnMl=sklearnMl, Xdf=Xdf, ydf=None)
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearn_model_crossvalidation(sklearnMl=sklearnMl, Xdf=Xdf, ydf=ydf, cv=4.5)
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearn_model_crossvalidation(sklearnMl=sklearnMl, Xdf=Xdf, ydf=ydf, scoring="BT")
    with pytest.raises(InvalidParameterValueException):
        sklearn_model_crossvalidation(sklearnMl=sklearnMl, Xdf=Xdf, ydf=ydf, scoring=["BT", "AT"])
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearn_model_crossvalidation(sklearnMl=sklearnMl, Xdf=Xdf, ydf=ydf, n_jobs=["BT", "AT"])
    with pytest.raises(BeartypeCallHintParamViolation):
        sklearn_model_crossvalidation(sklearnMl=sklearnMl, Xdf=Xdf, ydf=ydf, pre_dispatch=0.9)


test_sklearn_model_crossvalidation()
test_sklearn_model_crossvalidation_error()
