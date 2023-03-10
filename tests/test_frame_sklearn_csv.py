
from pathlib import Path
import sys

scripts = r'#/eis_toolkit/s'
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

##############################
# example for CSV - data:   
#            Task:   find the right location of a sample (given by analytics)

# ##############################################################################################
# A: Training
#

# #############################
# 1. Input - Interface: for training 
#                        to create by EIS-GUI
#                        Data (csv) and Fields-Dictionary 
#                                       field-name and field-type: i - identification, t target, v - values (float/integer), c - category, n - not to use )
#
#    
parent_dir = Path(__file__).parent
name_csv = str(parent_dir.joinpath(r'data/csv/Trainings_Test.csv'))

#name_csv_testset = str(parent_dir.joinpath(r'data/csv/Test_Test.csv'))
#name_fc = parent_dir.joinpath(r'data/csv/rwe_Testdatensatz.csv')

fields=  {'LfdNr':'i','Tgb':'n','TgbNr':'t','SchneiderThiele':'n','SuTNr':'n','Asche_trocken':'v','C_trocken':'n','H_trocken':'v',
       'S_trocken':'v','N_trocken':'v','AS_Natrium_ppm':'v','AS_Kalium_ppm':'v','AS_Calcium_ppm':'v',
       'AS_Magnesium_ppm':'v','AS_Aluminium_ppm':'v','AS_Silicium_ppm':'v','AS_Eisen_ppm':'v','AS_Titan_ppm':'v','AS_Schwefel_ppm':'n',
       'H_S':'v','Na_H':'v','K_H':'v','Ca_H':'v','Mg_H':'v','Al_H':'v','Si_H':'v','Fe_H':'v','Ti_H':'v',
       'Na_S':'v','K_S':'v','Ca_S':'v','Mg_S':'v','AL_S':'v','Si_S':'v','Fe_S':'v',
       'Ti_S':'v','Na_K':'v','Na_Ca':'v','Na_Mg':'v','Na_Al':'v','Si_Na':'v','Na_Fe':'v','Na_Ti':'v','K_Ca':'v','K_Mg':'v',
       'K_Al':'v','Si_K':'v','K_Fe':'v','K_Ti':'v','Ca_Mg':'v','Ca_Al':'v',
       'Si_Ca':'v','Ca_Fe':'v','Ca_Ti':'v','Mg_Al':'v','Si_Mg':'v','Mg_Fe':'v','Mg_Ti':'v','Si_Al':'v',
       'Al_Fe':'v','Al_Ti':'v','Si_Fe':'v','Si_Ti':'v','Fe_Ti':'v'}

#########################################
# 2. Import of csv-data based on fields-dictionary

print ('+++++++++++++++++++++++++++++++++++++++++    Part 1: Import')
columns , df , urdf , metadata = import_featureclass(fields = fields , file = name_csv , decimalpoint_german = True) 
# columns:  field-directory for all imported fields (= columns in df)
# df: DataFrame with the columns imported (fields with type i, t, v and c)
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
Xvdf , Xcdf , ydf , igdf = separation(df = df, fields = columns)
# Xvdf: DataFrame with all value columns whitch are not catagories (if no v-field: Xvdf is None)
# Xcdf: DataFrame with all catagories (if no c-field: Xcdf is None)
# ydf: DataFrame for the target column (one column, e.g. Location)
# igdf: DataFrame of (one) identifier-column and (if exists) the geometry column (if no i- and g - field igdf is None)

# 3.3: nodata replacement (if nodata exists and no nodata removal is used)
# rtypes:
# - 'replace'        argument replacement integer applicabel
# - 'mean'           applicable for values (not categories)  
# - 'median'         applicable for values 
# - 'n_neighbors'    argument k_neighbors (int) applicabel for the numbers of neighbors
# - 'most_frequent'  applicable for categories

Xvdf = nodata_replace(df = Xvdf, rtype = 'mean')
ydf = nodata_replace(df = ydf, rtype = 'most_frequent')
       # in case of classification - model: 'most_frequent' 
       # (in ydf should not be a nodata cell)
Xcdf = nodata_replace(df = Xcdf, rtype = 'most_frequent')  #, replacement_string = 'replace',replacement_number = 0) 
      

# 3.4: onehotencoding if catagories fields (c - fields) exists in imported data
Xdfneu1 , enc1 = onehotencoder(df = Xcdf)
# Xdfneu1: DataFrame with binariesed categorical columns
# enc1: object of the onehotencoder, this will be used for the imported data in prediction process 

# 3.5: putting binarised Xcdf and Xvdf together again
Xdf = unification(Xvdf = Xvdf, Xcdf = Xdfneu1)

#########################################
# 4. Creating of the sklearn model (random forest or logistic regression)

sklearnMl = sklearn_randomforest_classifier (oob_score = True)
#sklearnMl = sklearn_logistic_regression() 
#sklearnMl = sklearn_randomforest_regressor(oob_score=True)
# special arguments are applicable

#########################################
# 5. Validation of the sklearn model 

print ('+++++++++++++++++++++++++++++++++++++++++    Part 3: Validation')

# 5.1: Splitting of the dataset 
#      only to use if a testset is needed for validation
# train_X, test_X, train_y, test_y = all_split (Xdf = Xdf, ydf = ydf, test_size = 0.2)
       # trin_X, train_y for Training
       # test_X, test_y for Validation known y-values

# 5.2: validation with testset of 20% of the Trainingsdata
validation , confusion , comparison , sklearnMl = sklearn_model_validations (sklearnMl = sklearnMl , Xdf = Xdf , ydf = ydf , comparison = True , confusion_matrix = True , test_size = 0.2)
# validation (dictionary), confusion(DataFrame), comparison (list):  result of  validation will be saved to csv and json-files with all_export_files)
# sklearnML:  a fitted model based on the trainings dataset (80% of Xdf)

# 5.3 calculation of the importance of the columns of X (parmeter of the model, permutation importance, very time consuming)
#importance = sklearn_model_importance(sklearnMl= sklearnMl, Xdf=Xdf, ydf = ydf, n_repeats=5)
# importance: DataFrame  (very time consuming)
importance = sklearn_model_importance(sklearnMl= sklearnMl)
# importance: DataFrame  for random forest: faster

# 5.4 crossvalidation: (4 parts, 4 models, very time consuming)
cv = sklearn_model_crossvalidation (sklearnMl = sklearnMl, Xdf = Xdf, ydf = ydf, cv = 4)
# cv:  Crossvalidation (dictionary)

# 5.5. Export of the validation results 
path = str(parent_dir.joinpath(r'data'))
filesdict = export_files(name = 'test_csv',
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

print ('+++++++++++++++++++++++++++++++++++++++++    Teil 4: Fiting of the model')

# 6.1 Training
sklearnMl = sklearn_model_fit(sklearnMl = sklearnMl, Xdf = Xdf, ydf = ydf) #, fields = columns)

# 6.2 Export of the model, ohe-object, and fields
#      not necessary if the prediction part is following just no 

path = str(parent_dir.joinpath(r'data'))                                 
filesdict = export_files(name = 'test_csv',
       path = path,
       sklearnMl = sklearnMl,
       myFields = columns,
       skleanrOhe = enc1,
       new_version = False,
       decimalpoint_german = True,
)
# filesdict: Dictionary of the name of the files:   Interface for EIS-programm (GUI)

#########################################
# 7. Import  for Prediction
print ('+++++++++++++++++++++++++++++++++++++++++    Part 5: Import and Prepataion for Prediction')

# 7.1 import model, ohe-object and fields
sklearnMlp, myOhep, myFieldsp, myMetada, kerasMl, kerasOhe = import_files(
       sklearnMl_file = filesdict['sklearnMl'], 
       myFields_file = filesdict['myFields'], 
       sklearnOhe_file = filesdict['sklearnOhe'],
)

# 7.2 import csv
name_csvp = parent_dir.joinpath(r'data/csv/Test_Test.csv')

fields, df, urdf= import_featureclass(fields = myFieldsp, file = name_csvp, decimalpoint_german = True)


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
Xdf  = sklearn_check_prediction(sklearnMl=sklearnMlp, Xdf  = Xdf)
ydf1 = sklearn_model_prediction(sklearnMl = sklearnMlp,Xdf = Xdf)

print ('+++++++++++++++++++++++++++++++++++++++++    Part 7: Probability')
ydf2 = sklearn_model_predict_proba(sklearnMl = sklearnMlp,Xdf = Xdf, igdf=igdf)


# # #################### ydf an XDF anfügen

# Ergebnis = export_featurclass(Xdf = Xdf, df = ydf, nanmask = nodatmask)
parent_dir = Path(__file__).parent.__str__()+'/Ergebnis'
name_fc = 'csv_Ergebnis'
ext = '.csv'
Ergebnis = export_featureclass(ydf = ydf1, dfg = urdf, outpath = parent_dir.__str__(), outfile = name_fc, outextension = ext,decimalpoint_german=True, nanmask = nodatmask)

name_fc ='csv_Ergebnis_Wahrscheinlichkeit'
ext = 'csv'
#name_fc = parent_dir.joinpath(r'data/Ergebnis.shp')
Ergebnis_predict = export_featureclass(ydf = ydf2, outpath = parent_dir.__str__(), outfile = name_fc,outextension = ext,) # nanmask = nodatmask,decimalpoint_german=True,)  #,outextension = ext,, decimalpoint_german=True

print('=========== Ergebnis ===============:' )
# print(Ergebnis)

# parent_dir = Path(__file__).parent
# name_fc = parent_dir.joinpath(r'data/Ergebnis_1.shp')
# Ergebnis.to_file(name_fc)

print ('+++++++++++++++++++++++++++++++++++++++++    Part 8: Vergleich mit bekannten Ergebnis')
# extenes Testset

validation,confusion,comparison,sklearnMl = sklearn_model_validations(sklearnMlp, ydf = yt, predict_ydf=ydf1, comparison= True) #, fields = columns) #, test_size = 0.2)

filesdict = export_files(name = 'test_rwe_datenverglich',
       validations = validation, 
       confusion_matrix = confusion, 
       comparison = comparison,
       # crossvalidation = cv,
       # sklearnMl = sklearnMl, 
       # myFields = columns, 
       # myOhe = enc1,
       # myMetadata = Metadata,
       decimalpoint_german = True,
       new_version = False,
)
