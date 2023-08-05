from pathlib import Path

import pandas as pd
import pytest
# from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation
from sklearn.preprocessing import OneHotEncoder

# from eis_toolkit.checks.sklearn_check_prediction import sklearn_check_prediction
# from eis_toolkit.conversions.export_featureclass import export_featureclass
# from eis_toolkit.conversions.export_grid import export_grid
from eis_toolkit.conversions.import_featureclass import import_featureclass
# from eis_toolkit.conversions.import_grid import import_grid
# from eis_toolkit.exceptions import (InvalidParameterValueException)  # FileWriteError, FileReadError)
from eis_toolkit.file.export_files import export_files
from eis_toolkit.file.import_files import import_files
from eis_toolkit.prediction.sklearn_model_fit import sklearn_model_fit
# from eis_toolkit.prediction.sklearn_model_predict_proba import sklearn_model_predict_proba
from eis_toolkit.prediction.sklearn_randomforest_classifier import sklearn_randomforest_classifier
from eis_toolkit.transformations.nodata_replace import nodata_replace
from eis_toolkit.transformations.onehotencoder import onehotencoder
from eis_toolkit.transformations.separation import separation
from eis_toolkit.transformations.split import split
from eis_toolkit.transformations.unification import unification

# scripts = r"/eis_toolkit"  # /eis_toolkit/conversions'
# sys.path.append(scripts)

# from eis_toolkit.prediction.sklearn_model_prediction import sklearn_model_prediction

# from eis_toolkit.prediction.sklearn_randomforest_regressor import sklearn_randomforest_regressor

# from eis_toolkit.validation.sklearn_model_crossvalidation import sklearn_model_crossvalidation
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

# columns , df , urdf , metadata = import_featureclass(fields = fields_fc , file = name_fc , layer = layer_name)
columns, df, urdf, metadata = import_featureclass(fields=fields_csv, file=name_csv, decimalpoint_german=True)
# nodata_remove

# split
Xdf_train, Xdf_test, ydf_train, ydf_test = split(Xdf=df, test_size=10)

# Separation
Xvdf, Xcdf, ydf, igdf = separation(df=df, fields=columns)
# nodata_replacement of
Xcdf = nodata_replace(df=Xcdf, rtype="most_frequent")
Xvdf = nodata_replace(df=Xvdf, rtype="mean")
# onehotencoder
Xdf_enh, eho = onehotencoder(df=Xcdf)
# unification
Xdf = unification(Xvdf=Xvdf, Xcdf=Xdf_enh)
# model
sklearnMl = sklearn_randomforest_classifier(oob_score=True)
# fit
sklearnMl = sklearn_model_fit(sklearnMl=sklearnMl, Xdf=Xdf, ydf=ydf)
# export files
parent_dir = Path(__file__).parent.parent
path = str(parent_dir.joinpath(r"data"))
filesdict = export_files(
    name="test_csv",
    path=path,
    sklearnMl=sklearnMl,
    sklearnOhe=eho,
    myFields=columns,
    decimalpoint_german=True,
    new_version=False,
)

# import files
sklearnMlp, sklearnOhep, myFieldsp = import_files(
    sklearnMl_file=filesdict["sklearnMl"],
    sklearnOhe_file=filesdict["sklearnOhe"],
    myFields_file=filesdict["myFields"],
)

Xvdf_test, Xcdf_test, ydf_dmp, igdf_test = separation(df=Xdf_test, fields=myFieldsp)
# nodata_replacement of
Xcdf_test = nodata_replace(df=Xcdf_test, rtype="most_frequent")
# nodata_replacement of
Xvdf_test = nodata_replace(df=Xvdf_test, rtype="mean")


#################################################################


def test_onehotencoder():
    """Test functionality of onehotencoder of categorized columns of  imported X (Dataframe)."""
    Xdf_enh, eho = onehotencoder(df=Xcdf, ohe=sklearnOhep)

    assert (isinstance(eho, OneHotEncoder)) or (eho is None)
    assert (isinstance(Xdf_enh, pd.DataFrame)) or (Xcdf is None)
    if (Xdf_enh is not None) and (Xcdf is not None):
        assert len(Xcdf.index) == len(Xdf_enh.index)


def test_onehotencoder_error():
    """Test wrong arguments of onehotencoder with wron arguments."""
    with pytest.raises(BeartypeCallHintParamViolation):
        onehotencoder(df={"a": "A"})
    with pytest.raises(BeartypeCallHintParamViolation):
        onehotencoder(df=Xcdf, ohe=["i"])


test_onehotencoder()
test_onehotencoder_error()
