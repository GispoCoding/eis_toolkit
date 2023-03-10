

from pathlib import Path
import sys

scripts = r'/eis_toolkit'  #/eis_toolkit/conversions'
sys.path.append (scripts)

from eis_toolkit.conversions.export_featureclass import *
from eis_toolkit.conversions.import_featureclass import *
from eis_toolkit.transformations.nodata_replace import *
from eis_toolkit.transformations.nodata_remove import *
from eis_toolkit.transformations.separation import *
from eis_toolkit.transformations.split import *
from eis_toolkit.file.export_files import *
from eis_toolkit.file.import_files import *
from eis_toolkit.transformations.onehotencoder import *
from eis_toolkit.transformations.unification import *
from eis_toolkit.checks.sklearn_check_prediction import *
from eis_toolkit.model_training.sklearn_randomforest_regressor import *
from eis_toolkit.model_training.sklearn_randomforest_classifier import *
from eis_toolkit.model_training.sklearn_logistic_regression import *
from eis_toolkit.prediction_methods.sklearn_model_prediction import *
from eis_toolkit.prediction_methods.sklearn_model_predict_proba import *
from eis_toolkit.validation.sklearn_model_crossvalidation import *
from eis_toolkit.model_training.sklearn_model_fit import *
from eis_toolkit.validation.sklearn_model_validations import *
from eis_toolkit.validation.sklearn_model_importance import *


#########################################
# Excample Featureclass (gpkg-layer): 

# ##############################################################################################
# A: Training
#

# #############################
# 1. Input - Interface: for training 
#                        to create by EIS-GUI
#                        Featureclass-data (gdkg-layer) and Fields-Dictionary 
#                                       field-name and field-type: i - identification, t target, v - values (float/integer), c - category, g - geometry, n - not to use )
# 

parent_dir = Path(__file__).parent
name_fc = str(parent_dir.joinpath(r'data/shps/EIS_gp.gpkg'))
layer_name = r'Occ_2'

fields=  {'OBJECTID':'i', 'ID':'n', 'Name':'n', 'Alternativ':'n', 'Easting_EU':'n', 'Northing_E':'n',
       'Easting_YK':'n', 'Northing_Y':'n', 'Commodity':'c', 'Size_by_co':'c', 'All_main_c':'n',
       'Other_comm':'n', 'Ocurrence_':'n', 'Mine_statu':'n', 'Discovery_':'n', 'Commodity_':'n',
       'Resources_':'n', 'Reserves_t':'n', 'Calc_metho':'n', 'Calc_year':'n', 'Current_ho':'n',
       'Province':'n', 'District':'n', 'Deposit_ty':'n', 'Host_rocks':'n', 'Wall_rocks':'n',
       'Deposit_sh':'c', 'Dip_azim':'v', 'Dip':'t', 'Plunge_azi':'v', 'Plunge_dip':'v', 'Length_m':'v',
       'Width_m':'n', 'Thickness_':'v', 'Depth_m':'n', 'Date_updat':'n', 'geometry':'g'}

#########################################
# 2. Import of fc-data based on fields-dictionary

print ('+++++++++++++++++++++++++++++++++++++++++    Part 1: Import')
columns, df, urdf, metadata= import_featureclass(fields = fields, file = name_fc, layer = layer_name)
# columns:  field-directory for all imported fields (= columns in df)
# df: DataFrame with the columns imported (fields with type i, g, t, v and c)
# urdf: DataFrame with all imported columns, the will be addd with target column in case of prediction
# metadata: None for csv (importend for featureclasses (crs) or images)


#########################################
# 3. either nodata replacement or nodata removement  (one of both) and onehotencoding

print ('+++++++++++++++++++++++++++++++++++++++++    Part 2: Preparation')
# 3.1: nodata removement 
# df_new, nodatmask = all_nodata_remove(df = df)
# # df_new:  Dataframe without rows containing one or more nodata cells
# # nodatamask: DataFrame with one column: True - if the row is removed (not stoed in df_new), False - if the row is in df_new

# 3.2: preparation for nodata replacemnt, onehotencoding and further training or prediction
Xvdf, Xcdf, ydf, igdf = separation(df = df, fields = fields)
# Xvdf: DataFrame with all value columns whitch are not catagories (if no v-field: Xvdf is None)
# Xcdf: DataFrame with all catagories (if no c-field: Xcdf is None)
# ydf: DataFrame for the target column (one column, e.g. Location)
# igdf: DataFrame of (one) identifier-column and (if exists) the geometry column (if no i- and g - field igdf is None)

Xvdf = nodata_replace(df = Xvdf, rtype = 'replace', replacement_number = 99)
ydf = nodata_replace(df = ydf, rtype = 'most_frequent')
       # in case of classification - model: 'most_frequent' 
       # (in ydf should not be a nodata cell)
Xcdf = nodata_replace(df = Xcdf, rtype = 'most_frequent')

# 3.4: onehotencoding if catagories fields (c - fields) exists in imported data
Xdfneu1 , enc1 = onehotencoder(df = Xcdf)
# Xdfneu1: DataFrame with binariesed categorical columns
# enc1: object of the onehotencoder, this will be used for the imported data in prediction process 

# 3.5: putting binarised Xcdf and Xvdf together again
Xdf = unification(Xvdf = Xvdf, Xcdf = Xdfneu1)

#########################################
# 4. Creating of the sklearn model (random forest or logistic regression)

sklearnMl = sklearn_randomforest_classifier (oob_score = True)
# Xdfneu1: DataFrame with binariesed categorical columns
# enc1: object of the onehotencoder, this will be used for the imported data in prediction process 


#########################################
# 5. Validation of the sklearn model 

print ('+++++++++++++++++++++++++++++++++++++++++    Part 3: Validation')

# 5.1: Splitting of the dataset 
#      only to use if a testset is needed for validation
# train_X, test_X, train_y, test_y = all_split (Xdf = Xdf, ydf = ydf, test_size = 0.2)
       # trin_X, train_y for Training
       # test_X, test_y for Validation known y-values

# 5.2: validation with testset of 20% of the Trainingsdata
validation, confusion, comparison, sklearnMl = sklearn_model_validations(sklearnMl = sklearnMl, Xdf = Xdf, ydf = ydf, comparison = True, confusion_matrix = True, test_size = 0.2)
# validation (dictionary), confusion(DataFrame), comparison (list):  result of  validation will be saved to csv and json-files with all_export_files)
# sklearnML:  a fitted model based on the trainings dataset (0,8 -> 80% of Xdf)

# 5.3 calculation of the importance of the columns of X (parmeter of the model, permutation importance, very time consuming)
#importance = sklearn_model_importance(sklearnMl= sklearnMl, Xdf=Xdf, ydf = ydf, n_repeats=5)
# importance: DataFrame  (very time consuming)
importance = sklearn_model_importance(sklearnMl= sklearnMl,)
# importance: DataFrame  for random forest: faster

# 5.4 crossvalidation: (4 parts, 4 models, very time consuming)
cv = sklearn_model_crossvalidation(sklearnMl = sklearnMl, Xdf = Xdf, ydf = ydf, cv = 4)
# cv:  Crossvalidation (dictionary)


# 5.5. Export of the validation results 
path = str(parent_dir.joinpath(r'data'))
filesdict = export_files(name = 'test_fc',
       path = path,
       validations = validation,
       confusion_matrix = confusion,
       comparison = comparison,
       crossvalidation = cv,
       importance = importance,
       new_version = False,
       decimalpoint_german = True,
)
# filesdict: Dictionary of the name of the files:   Interface for EIS-programm (GUI)

#########################################
# 6. Fiting of the sklearn model 

print ('+++++++++++++++++++++++++++++++++++++++++   Part 4: Fiting of the model')

# 6.1 Training
sklearnMl = sklearn_model_fit(sklearnMl = sklearnMl, Xdf = Xdf, ydf = ydf)

# 6.2 Export of the model, ohe-object, and fields
#      not necessary if the prediction part is following just no 

filesdict = export_files(name = 'test_fc',
       validations = validation, 
       confusion_matrix = confusion, 
       comparison = comparison,
       importance = importance,
       new_version = False,
       decimalpoint_german = True,
)

path = str(parent_dir.joinpath(r'data'))                                 
filesdict = export_files(name = 'test_fc',
       path = path,
       sklearnMl = sklearnMl,
       myFields = columns,
       sklearnOhe = enc1,
       new_version = False,
       decimalpoint_german = True,
)
# filesdict: Dictionary of the name of the files:   Interface for EIS-programm (GUI)

#########################################
# 7. Import  for Prediction
print ('+++++++++++++++++++++++++++++++++++++++++    Part 5: Import and Prepataion for Prediction')

# 7.1 import model, ohe-object and fields
sklearnMlp,myOhep,myFieldsp,kerasMl,kerasOhe = import_files(
       sklearnMl_file = filesdict['sklearnMl'], 
       myFields_file = filesdict['myFields'], 
       sklearnOhe_file = filesdict['sklearnOhe'],
)

# 7.2 import fc
name_fcp = name_fc
columns, df, urdf,metadata= import_featureclass(fields = myFieldsp, file = name_fcp) #, layer = layer_name  ) #, decimalpoint_german = True)


#########################################
# 8. Prepataion for Prediction

df_pr, nodatmask = nodata_remove(df = df)
Xvdf, Xcdf, yt, igdf = separation(df = df_pr, fields = fields)

# Xvdf = nodata_replace(df = Xvdf, rtype = 'replace')
# Xcdf = nodata_replace(df = Xcdf, rtype = 'most_frequent')
# ydf = nodata_replace(df = t, rtype = 'most_frequent')

Xdfneu2, enc2 = onehotencoder(df = Xcdf, ohe = myOhep)

Xdf = unification(Xvdf = Xvdf, Xcdf = Xdfneu2)

print ('+++++++++++++++++++++++++++++++++++++++++    Part 6: Prediction')
Xdf  = sklearn_check_prediction(sklearnMlp,Xdf,)
ydf1 = sklearn_model_prediction(sklearnMl = sklearnMlp, Xdf = Xdf,)

print ('+++++++++++++++++++++++++++++++++++++++++    Part 7: Probability')
ydf2 = sklearn_model_predict_proba(sklearnMl = sklearnMlp, Xdf = Xdf, igdf=igdf, fields = myFieldsp)


# # #################### ydf an XDF anfügen

# Ergebnis = export_featurclass(Xdf = Xdf, df = ydf, nanmask = nodatmask)
parent_dir = Path(__file__).parent.__str__()+r'/data/shps/EIS.gpkg'
name_fc ='fc_Ergebnis'
ext = ''
#name_fc = parent_dir.joinpath(r'data/Ergebnis.shp')
Ergebnis = export_featureclass(ydf = ydf1, dfg = urdf, metadata = metadata,outpath = parent_dir.__str__(), outfile = name_fc, nanmask = nodatmask)

parent_dir = Path(__file__).parent.__str__()+r'/data/shps/EIS.gpkg'
name_fc ='fc_Ergebnis_Wahrscheinlichkeit'
ext = ''
#name_fc = parent_dir.joinpath(r'data/Ergebnis.shp')
Ergebnis_predict = export_featureclass(ydf = ydf2, metadata = metadata, outpath = parent_dir.__str__(), outfile = name_fc)

print('=========== Ergebnis ===============:' )
