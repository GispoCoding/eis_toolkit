########################
# Test frame for single modules
#

from pathlib import Path

# import matplotlib.pyplot as plt
# from rasterio.enums import Resampling
# from rasterio.plot import show

# scripts = r"/eis_toolkit"  # /eis_toolkit/conversions'
# sys.path.append(scripts)

from eis_toolkit.checks.sklearn_check_prediction import sklearn_check_prediction
# from eis_toolkit.conversions.export_featureclass import export_featureclass
from eis_toolkit.conversions.export_grid import export_grid
# from eis_toolkit.conversions.import_featureclass import import_featureclass
from eis_toolkit.conversions.import_grid import import_grid
# from eis_toolkit.exceptions import (InvalidParameterValueException)  # FileWriteError, FileReadError)
from eis_toolkit.file.export_files import export_files
from eis_toolkit.file.import_files import import_files
from eis_toolkit.prediction.sklearn_model_fit import sklearn_model_fit
from eis_toolkit.prediction.sklearn_model_prediction import sklearn_model_prediction
# from eis_toolkit.prediction.sklearn_model_predict_proba import sklearn_model_predict_proba
# from eis_toolkit.prediction.sklearn_randomforest_classifier import sklearn_randomforest_classifier
from eis_toolkit.prediction.sklearn_randomforest_regressor import sklearn_randomforest_regressor
from eis_toolkit.transformations.nodata_remove import nodata_remove
# from eis_toolkit.transformations.nodata_replace import nodata_replace
from eis_toolkit.transformations.onehotencoder import onehotencoder
from eis_toolkit.transformations.separation import separation
# from eis_toolkit.transformations.split import split
from eis_toolkit.transformations.unification import unification
from eis_toolkit.validation.sklearn_model_crossvalidation import sklearn_model_crossvalidation
from eis_toolkit.validation.sklearn_model_importance import sklearn_model_importance
from eis_toolkit.validation.sklearn_model_validations import sklearn_model_validations

rname = "test_images"  # name of the project
parent_dir = Path(__file__).parent
print("*****************************************************")
# Aktuell Testrahmen des Einlesens eines GRID
# Dateiname eines GRID:

#############################
# example for image - data:
#            Task:   data from Gold ghana

# ##############################################################################################
# A: Training
#

# #############################
# 1. Input - Interface: for training
#                        Data (tif) and Fields-Dictionary
#

# name_K = str(parent_dir.joinpath(r'data/Primary_data/Rad/IOCG_Gm_Rd_K_.tif'))
# name_Th = str(parent_dir.joinpath(r'data/Primary_data/Rad/IOCG_Gm_Rd_Th_eq_.tif'))
# name_U = str(parent_dir.joinpath(r'data/Primary_data/Rad/IOCG_Gm_Rd_U_eq_.tif'))
# name_target = str(parent_dir.joinpath(r'data/Primary_data/Rad/IOCG_Gm_Rd_Total_Count_.tif'))

# #  columns-name    column-type  file-name    
#  type: 
#      i - identification, 
#      t target, 
#      v - values (float/integer), 
#      c - category, 
#      n - not to use )

# grids =  [{'name':'Total','type':'t','file':name_target},
#  {'name':'Kalium', 'file':name_K, 'type':'v'},
#  {'name':'Thorium', 'file':name_Th, 'type':'v'},
#  #{'name':'Uran', 'file':name_U, 'type':'v'}]
# ]

# Ghana
# name_tif1 = parent_dir.joinpath(r'data/Ghana/Deposits/gold_dep100/dblbnd.adf')  # nodata = -1 wird nicht genommen
# name_tif1 = str(parent_dir.joinpath(r'data/Ghana/deposits.tif'))
# name_tif2 = str(parent_dir.joinpath(r'data/Ghana/em_hfif_gy_asp_sn_s.tif'))
# name_tif3 = parent_dir.joinpath(r'data/Ghana/gy_scaled.tif')
# # name_tif4 = parent_dir.joinpath(r'data/Ghana/Extr_a2_em_hfrf.tif')
# # name_tif5 = parent_dir.joinpath(r'data/Ghana/Extr_a2_em_lfif.tif')
# # name_tif6 = parent_dir.joinpath(r'data/Ghana/Extr_a2_mag_ano.tif')
# # name_tif7 = parent_dir.joinpath(r'data/Ghana/Extr_a2_mag_mtlev.tif')
# # name_tif8 = parent_dir.joinpath(r'data/Ghana/Extr_a2_spc_kal.tif')
# # name_tif9 = parent_dir.joinpath(r'data/Ghana/Extr_a2_spc_tho.tif')
# # name_tif10 = parent_dir.joinpath(r'data/Ghana/Extr_a2_spc_tot.tif')
# # name_tif11 = parent_dir.joinpath(r'data/Ghana/Extr_a2_spc_ura.tif')


# # {'name':'em_hfrf', 'file':name_tif4, 'type':'v'},
# # {'name':'em_lfif', 'file':name_tif5, 'type':'v'},
# # {'name':'mag_ano', 'file':name_tif6, 'type':'v'},
# # {'name':'mag_mtlev', 'file':name_tif7, 'type':'v'},
# # {'name':'spc_kal', 'file':name_tif8, 'type':'v'},
# # {'name':'spc_tho', 'file':name_tif9, 'type':'v'},
# # {'name':'spc_tot', 'file':name_tif10, 'type':'v'},
# # {'name':'spc_ura', 'file':name_tif11, 'type':'v'}
# ]

# parent_dir = Path(__file__).parent.parent.parent.joinpath(r'Daten/Ghana_Gold_NW_04_ModelInputData/')
# #name_tif1 = parent_dir.joinpath(r'Deposits/gold_dep100/dblbnd.adf')  # nodata = -1 wird nicht genommen
# name_tif1 = str(parent_dir.joinpath(r'Deposits/deposits.tif'))
# name_tif2 = str(parent_dir.joinpath(r'EM_HFIF/gy_asp_sn_s/dblbnd.adf'))
# name_tif3 = str(parent_dir.joinpath(r'EM_HFIF/gy_asp_we_s/dblbnd.adf'))
# name_tif4 = str(parent_dir.joinpath(r'EM_HFIF/gy_scaled.tif'))
# name_tif5 = str(parent_dir.joinpath(r'EM_HFIF/gy_slp_s.tif'))

# name_tif6 = str(parent_dir.joinpath(r'EM_HFRF/gy_asp_sn_s.tif'))
# name_tif7 = str(parent_dir.joinpath(r'EM_HFRF/gy_asp_we_s.tif'))
# name_tif8 = str(parent_dir.joinpath(r'EM_HFRF/gy_scaled.tif'))
# name_tif9 = str(parent_dir.joinpath(r'EM_HFRF/gy_slp_s.tif'))

# name_tif10 = str(parent_dir.joinpath(r'EM_LFIF/gy_asp_sn_s.tif'))
# name_tif11 = str(parent_dir.joinpath(r'EM_LFIF/gy_asp_we_s.tif'))
# name_tif12 = str(parent_dir.joinpath(r'EM_LFIF/gy_scaled.tif'))
# name_tif13 = str(parent_dir.joinpath(r'EM_LFIF/gy_slp_s.tif'))

# name_tif14 = str(parent_dir.joinpath(r'EM_LFRF/gy_asp_sn_s.tif'))
# name_tif15 = str(parent_dir.joinpath(r'EM_LFRF/gy_asp_we_s.tif'))
# name_tif16 = str(parent_dir.joinpath(r'EM_LFRF/gy_scaled.tif'))
# name_tif17 = str(parent_dir.joinpath(r'EM_LFRF/gy_slp_s.tif'))

# name_tif18 = str(parent_dir.joinpath(r'geo2/a2_tec1_dist.tif'))
# name_tif19 = str(parent_dir.joinpath(r'geo2/a2_tec2_dist.tif'))
# name_tif20 = str(parent_dir.joinpath(r'geo2/a2_tec3_dist.tif'))
# name_tif21 = str(parent_dir.joinpath(r'geo2/a2_tecc_dist.tif'))

# name_tif18 = str(parent_dir.joinpath(r'geo2/a2_tec1_dist.tif'))
# name_tif19 = str(parent_dir.joinpath(r'geo2/a2_tec2_dist.tif'))
# name_tif20 = str(parent_dir.joinpath(r'geo2/a2_tec3_dist.tif'))
# name_tif21 = str(parent_dir.joinpath(r'geo2/a2_tecc_dist.tif'))


# grids=  [{'name':'Deposit','type':'t','file':name_tif1},
#  {'name':'em_hfif_gy_asp_sn_s', 'file':name_tif2, 'type':'v'},
#  {'name':'gy_scaled', 'file':name_tif3, 'type':'v'},
# ]

name_tif1 = str(parent_dir.joinpath(r"data/test1.tif"))
name_tif2 = str(parent_dir.joinpath(r"data/test2.tif"))
name_tif3 = parent_dir.joinpath(r"data/test1.tif")

grids = [
    {"name": "targe", "type": "t", "file": name_tif1},
    {"name": "test1", "file": name_tif2, "type": "v"},
    {"name": "test2", "file": name_tif3, "type": "v"},
]

print("+++++++++++++++++++++++++++++++++++++++++   Part 1 Import")

#########################################
# 2. Import of tif-data based on grids-dictionary

columns, df, metadata = import_grid(grids)
# columns:  fields-directory for all imported fields (= columns in df)
# df: DataFrame with the columns imported (fields with type b, t, v and c)
# metadata: None for csv (for featureclasses (crs) or images: crs, height, width...)

#########################################
# 3. either nodata replacement or nodata removement  (one of both) and onehotencoding

print("+++++++++++++++++++++++++++++++++++++++++    Part 2: Preparation")
# 3.1: nodata removement
df, nodatmask = nodata_remove(
    df=df,
)
#  df_new:  Dataframe without rows containing one or more nodata cells
#  nodatamask: DataFrame with one column: True - if the row is removed (not stoed in df_new), 
#                                          False - if the row is in df_new
#  Not nessesary for training

# 3.2: separation: preparation for nodata replacement, onehotencoding and further training or prediction
Xvdf, Xcdf, ydf, igdf = separation(
    df=df,
    fields=columns,
)
# Xvdf: DataFrame with all value columns whitch are not catagories (if no v-field: Xvdf is None)
# Xcdf: DataFrame with all catagories (if no c-field: Xcdf is None)
# ydf: DataFrame for the target column (one column, e.g. Location)
# igdf: DataFrame of (one) identifier and geometry-column: for images: None

# 3.3: nodata replacement (if nodata exists and no nodata removal is used): not for images

# 3.4: onehotencoding if catagories fields (c - fields) exists in imported data
Xdfneu, ohe = onehotencoder(
    df=Xcdf,
)
# Xdfneu1: DataFrame with binariesed categorical columns
# ohe: object of the onehotencoder, this will be used for the imported data in prediction process

# 3.5: putting binariesed Xcdf and Xvdf together again
Xdf = unification(
    Xvdf=Xvdf,
    Xcdf=Xdfneu,
)

#########################################
# 4. Creating of the sklearn model (random forest or logistic regression)

sklearnMl = sklearn_randomforest_regressor(
    oob_score=True,
)
# sklearnMl = sklearn_randomforest_classifier (oob_score = True)
# special arguments are applicable


#########################################
# 5. Validation of the sklearn model
print("+++++++++++++++++++++++++++++++++++++++++    Part 3: Validation")

# 5.1: Splitting of the dataset
#      only to use if a testset is wanted for validation
# train_X, test_X, train_y, test_y = split (Xdf = Xdf, ydf = ydf, test_size = 0.3,)
# train_X, train_y for Training
# test_X, test_y for validation known y-values
# smaller datast e.g. for importances
# print('split')
# 5.2: validation with testset of 20% of the Trainingsdata
validation, confusion, comparison, sklearnMl = sklearn_model_validations(
    sklearnMl=sklearnMl,
    Xdf=Xdf,
    ydf=ydf,
    test_size=0.1,
)
# validation, confusion, comparison, sklearnMl = sklearn_model_validations(sklearnMl=sklearnMl,
#                                                Xdf=train_X, ydf=train_y, test_size=0.1,)
# validation (dictionary), confusion(DataFrame), comparison (list):  
#        result of  validation will be saved to csv and json-files with all_export_files)
# sklearnML:  a fitted model based on the trainings dataset (80% of Xdf)
print("validation")

# 5.3 calculation of the importance of the columns of X 
#         (parmeter of the model, permutation importance, very time consuming)
# importance = sklearn_model_importance(sklearnMl= sklearnMl, Xdf=test_X, ydf = test_y, n_repeats=5)
# importance: DataFrame (very time consuming)

importance = sklearn_model_importance(
    sklearnMl=sklearnMl,
)
# importance: DataFrame  for random forest: faster
print("importance")

# 5.4 crossvalidation: (4 parts, 4 models, very time consuming)
cv = sklearn_model_crossvalidation(
    sklearnMl=sklearnMl,
    Xdf=Xdf,
    ydf=ydf,
    cv=3,
)
# cv:  Crossvalidation (dictionary)
print("cv")

# 5.5 Export of the validation results
path = str(parent_dir.joinpath(r"data"))
filesdict = export_files(
    name=rname,
    path=path,
    validations=validation,
    confusion_matrix=confusion,
    comparison=comparison,
    crossvalidation=cv,
    importance=importance,
    new_version=False,
    decimalpoint_german=True,
)
# filesdict: Dictionary of the name of the files:   Interface for EIS-programm (GUI)

#########################################
# 6. Fiting of the sklearn model

print("+++++++++++++++++++++++++++++++++++++++++    Teil 4: Fiting of the model")

sklearnMl = sklearn_model_fit(
    sklearnMl=sklearnMl,
    Xdf=Xdf,
    ydf=ydf,
)  # fields = fields,)

# 6.2 Export of the model, ohe-object and fields
#      not necessary if the prediction part is following just no
path = str(parent_dir.joinpath(r"data"))
filesdict = export_files(
    name=rname,
    path=path,
    sklearnMl=sklearnMl,
    myFields=columns,
    sklearnOhe=ohe,
)
# filesdict: Dictionary of the name of the files:   Interface for EIS-programm (GUI)

print("+++++++++++++++++++++++++++++++++++++++++    Prediction")

# #############################
# 7. Input - Interface: for training
#                        Data (tif) and Fields-Dictionary
#

# ame_tif1 = str(parent_dir.joinpath(r'data/Ghana/deposits.tif'))
# name_tif2 = str(parent_dir.joinpath(r'data/Ghana/em_hfif_gy_asp_sn_s.tif'))
# name_tif3 = parent_dir.joinpath(r'data/Ghana/gy_scaled.tif')
# # name_tif4 = parent_dir.joinpath(r'data/Ghana/Extr_a2_em_hfrf.tif')
# # name_tif5 = parent_dir.joinpath(r'data/Ghana/Extr_a2_em_lfif.tif')
# # name_tif6 = parent_dir.joinpath(r'data/Ghana/Extr_a2_mag_ano.tif')
# # name_tif7 = parent_dir.joinpath(r'data/Ghana/Extr_a2_mag_mtlev.tif')
# # name_tif8 = parent_dir.joinpath(r'data/Ghana/Extr_a2_spc_kal.tif')
# # name_tif9 = parent_dir.joinpath(r'data/Ghana/Extr_a2_spc_tho.tif')
# # name_tif10 = parent_dir.joinpath(r'data/Ghana/Extr_a2_spc_tot.tif')
# # name_tif11 = parent_dir.joinpath(r'data/Ghana/Extr_a2_spc_ura.tif')

# grids=  [
#  {'name':'em_hfif_gy_asp_sn_s', 'file':name_tif2, 'type':'v'},
#  {'name':'gy_scaled', 'file':name_tif3, 'type':'v'},
# ]
print("+++++++++++++++++++++++++++++++++++++++++    Part 1: Import of Model for Prediction")

# 8. Import of model, ohe-object and fields
sklearnMlp, myOhep, myFieldsp = import_files(
    sklearnMl_file=filesdict["sklearnMl"],
    myFields_file=filesdict["myFields"],
    sklearnOhe_file=filesdict["sklearnOhe"],
)

print("+++++++++++++++++++++++++++++++++++++++++    Part 2: Import for Prediction")

#########################################
# 9. Import of tif-data based on grids-dictionary
# columns, df, metadata = import_grid(grids)
# columns:  fields-directory for all imported fields (= columns in df)
# df: DataFrame with the columns imported (fields with type b, t, v and c)
# metadata: None for csv (for featureclasses (crs) or images: crs, height, width...)

print("+++++++++++++++++++++++++++++++++++++++++    Part 3: Preparation for Prediction")
#########################################
# 10. either nodata replacement or nodata removement  (one of both) and onehotencoding

# # 10.1: nodata removement
# df, nodatmask = nodata_remove(df = df)
#  df_new:  Dataframe without rows containing one or more nodata cells
#  nodatamask: DataFrame with one column: True 
#       - if the row is removed (not stoed in df_new), False - if the row is in df_new

# # 10.2: preparation for nodata replacement, onehotencoding and further training or prediction
# Xvdf, Xcdf, ydf, igdf = separation(df = Xdf, fields = columns)
# # Xvdf: DataFrame with all value columns whitch are not catagories (if no v-field: Xvdf is None)
# # Xcdf: DataFrame with all catagories (if no c-field: Xcdf is None)
# # (ydf: DataFrame for the target column (one column, e.g. Location): for prediction: None
# # (igdf: DataFrame of (one) identifier and geometry-column: for images: None )

# # 10.3: nodata replacement (if nodata exists and no nodata removal is used): not for images

# # 10.4: onehotencoding if catagories fields (c - fields) exists in imported data
# Xdfneu, ohe = onehotencoder(df = Xcdf, ohe = myOhep)
# # Xdfneu: DataFrame with binariesed categorical columns
# # ohe: object of the onehotencoder, this will be used from the imported data from training process

# # 10.5: putting binariesed Xcdf and Xvdf together again
# Xdf = unification(Xvdf = Xvdf, Xcdf = Xdfneu)


print("+++++++++++++++++++++++++++++++++++++++++    Part 4: Prediction")
#########################################
# 11. Check and Prediction

# 11.1: Check and reorder the fields of Xdf
Xdf = sklearn_check_prediction(
    sklearnMl=sklearnMlp,
    Xdf=Xdf,
)

# 11.2: Prediction
ydf1 = sklearn_model_prediction(
    sklearnMl=sklearnMlp,
    Xdf=Xdf,
)
# ydf2 = sklearn_model_predict_proba(sklearnMl = sklearnMlp,Xdf = Xdf)   #

print("+++++++++++++++++++++++++++++++++++++++++    Part 5: Output of the prediction result")
#########################################
# 12. Export

parent_dir = Path(__file__).parent.__str__() + r"/data"
name_fc = r"prediction_result_G"
ext = ".tif"
Ergebnis = export_grid(
    df=ydf1,
    metadata=metadata,
    outpath=parent_dir,
    outfile=name_fc,
    # outextension = ext,
    nanmask=nodatmask,
)

#
print("*****************************************************")
