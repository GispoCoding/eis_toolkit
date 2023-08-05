from pathlib import Path

import numpy as np
import pytest
# from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation

# scripts = r"/eis_toolkit"  # /eis_toolkit/conversions'
# sys.path.append(scripts)
from eis_toolkit.checks.sklearn_check_prediction import sklearn_check_prediction
from eis_toolkit.conversions.export_grid import export_grid
# from eis_toolkit.conversions.import_featureclass import import_featureclass
from eis_toolkit.conversions.import_grid import import_grid
from eis_toolkit.exceptions import FileWriteError  # InvalidParameterValueException,
from eis_toolkit.file.export_files import export_files
from eis_toolkit.file.import_files import import_files
from eis_toolkit.prediction.sklearn_model_fit import sklearn_model_fit
from eis_toolkit.prediction.sklearn_model_prediction import sklearn_model_prediction
# from eis_toolkit.prediction.sklearn_randomforest_classifier import sklearn_randomforest_classifier
from eis_toolkit.prediction.sklearn_randomforest_regressor import sklearn_randomforest_regressor
# from eis_toolkit.validation.sklearn_model_crossvalidation import sklearn_model_crossvalidation
# from eis_toolkit.validation.sklearn_model_importance import sklearn_model_importance
# from eis_toolkit.validation.sklearn_model_validations import sklearn_model_validations
# from eis_toolkit.prediction.sklearn_model_predict_proba import sklearn_model_predict_proba
from eis_toolkit.transformations.nodata_remove import nodata_remove
# from eis_toolkit.transformations.nodata_replace import nodata_replace
from eis_toolkit.transformations.onehotencoder import onehotencoder
from eis_toolkit.transformations.separation import separation
from eis_toolkit.transformations.split import split
from eis_toolkit.transformations.unification import unification

# import geopandas as gpd
# import pandas as pd

#################################################################
# import of grid: import_grid

# grid:
parent_dir = Path(__file__).parent.parent
name_K = str(parent_dir.joinpath(r"data/Primary_data/Rad/IOCG_Gm_Rd_K_.tif"))
name_Th = str(parent_dir.joinpath(r"data/Primary_data/Rad/IOCG_Gm_Rd_Th_eq_.tif"))
name_U = str(parent_dir.joinpath(r"data/Primary_data/Rad/IOCG_Gm_Rd_U_eq_.tif"))
name_target = str(parent_dir.joinpath(r"data/Primary_data/Rad/IOCG_Gm_Rd_Total_Count_.tif"))

# grids and grid-types for X (training based on tif-files)
grids = [
    {"name": "Total", "type": "t", "file": name_target},
    {"name": "Kalium", "file": name_K, "type": "v"},
    {"name": "Thorium", "file": name_Th, "type": "v"},
    {"name": "Uran", "file": name_U, "type": "v"},
]

name_tif1 = str(parent_dir.joinpath(r"data/test1.tif"))
name_tif2 = str(parent_dir.joinpath(r"data/test2.tif"))
name_tif3 = parent_dir.joinpath(r"data/test1.tif")

grids = [
    {"name": "targe", "type": "t", "file": name_tif1},
    {"name": "test1", "file": name_tif2, "type": "v"},
    {"name": "test2", "file": name_tif3, "type": "v"},
]

# Ghana:
deposits = str(parent_dir.joinpath(r"data/Ghana/deposits.tif"))
emhgas = str(parent_dir.joinpath(r"data/Ghana/em_hfif_gy_asp_sn_s.tif"))
gysc = str(parent_dir.joinpath(r"data/Ghana/gy_scaled.tif"))
tccr = str(parent_dir.joinpath(r"data/Ghana/tcnomax_crossings.tif"))

# Ghana
# grids =  [{'name':'Deposits','type':'t','file':deposits},
#  {'name':'em_h', 'file':emhgas, 'type':'v'},
#  {'name':'gy_scaled', 'file':gysc, 'type':'v'},
#  {'name':'tc_crossing', 'file':tccr, 'type':'v'}]

# import grids
columns, df, metadata = import_grid(grids)

# nodata_remove
df, nanmask = nodata_remove(df=df)
# split
Xdf_train, Xdf_test, ydf_train, ydf_test = split(Xdf=df, test_size=10)

# Separation
Xvdf, Xcdf, ydf, igdf = separation(df=df, fields=columns)
# nodata_replacement
# Xcdf = nodata_replace(df = Xcdf, rtype = 'most_frequent')
# Xvdf = nodata_replace(df = Xvdf, rtype = 'mean')
# ydf = nodata_replace(df = ydf, rtype = 'most_frequent')
# onehotencoder
Xdf_enh, eho = onehotencoder(df=Xcdf)
# unification
Xdf = unification(Xvdf=Xvdf, Xcdf=Xdf_enh)
# model
# sklearnMl = sklearn_randomforest_classifier(oob_score = True)
sklearnMl = sklearn_randomforest_regressor(oob_score=True)
# fit
sklearnMl = sklearn_model_fit(sklearnMl=sklearnMl, Xdf=Xdf, ydf=ydf)
# export files
parent_dir = Path(__file__).parent.parent
path = str(parent_dir.joinpath(r"data"))
filesdict = export_files(
    name="test_grid",
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

# # nodata_remove
# df_test,nanmask = nodata_remove(df = Xdf)
# # separation
# Xvdf_test, Xcdf_test, ydf_dmp, igdf_test = separation(df = Xdf, fields = myFieldsp)

# # onehotencoder
# Xdf_enht, eho = onehotencoder(df = Xcdf_test, ohe = sklearnOhep)
# # unification
# Xdf_tst = unification(Xvdf = Xvdf_test, Xcdf = Xdf_enht)
# check prediction
Xdf_pr = sklearn_check_prediction(sklearnMl=sklearnMlp, Xdf=Xdf)
# prediction
ydf_pr = sklearn_model_prediction(sklearnMl=sklearnMlp, Xdf=Xdf_pr)
# predict_proba


#################################################################


def test_export_grid():
    """Test functionality of export grid."""

    ydfpr = export_grid(df=ydf_pr, metadata=metadata, outpath=path, outfile="tst_image", nanmask=nanmask)
    assert isinstance(ydfpr, np.ndarray)   # pd.DataFrame)


def test_export_grid_error():
    """Test wrong arguments."""
    with pytest.raises(BeartypeCallHintParamViolation):
        export_grid(df=path, metadata=path,)
    with pytest.raises(BeartypeCallHintParamViolation):
        export_grid(df=ydf_pr, metadata=path, outpath=path, outfile="pr_image", nanmask=nanmask)

    path_wrong = str(parent_dir.joinpath(r"falsch"))
    with pytest.raises(FileWriteError):
        export_grid(df=ydf_pr, metadata=metadata, outpath=path_wrong, outfile="pr_image", nanmask=nanmask)
    with pytest.raises(BeartypeCallHintParamViolation):
        export_grid(df=ydf_pr, metadata=metadata, outpath=path, outfile=9.9, nanmask=nanmask)

    with pytest.raises(BeartypeCallHintParamViolation):
        export_grid(df=ydf_pr, metadata=metadata, outpath=path, outfile="pr_image", nanmask=",")


test_export_grid()
test_export_grid_error()
