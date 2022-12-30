'''
Bernd
'''

## for testing some code sniples
#import tensorflow as tf
# import numpy as np
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# x = tf.placeholder(shape=[None, 2], dtype=tf.float32)
# X_1 = tf.placeholder(tf.float32, name = "X_1")
# X_2 = tf.placeholder(tf.float32, name = "X_2")

# multiply = tf.multiply(X_1, X_2, name = "multiply")

# with tf.Session() as session:
#     result = session.run(multiply, feed_dict={X_1:[1,2,3], X_2:[4,5,6]})
#     print(result)

# t = 1


from pathlib import Path
import copy
import pandas as pd
import geopandas as gpd
import sys

# scripts = r'/eis_toolkit'  #/eis_toolkit/conversions'
# sys.path.append (scripts)

from export_featureclass import *
# from eis_toolkit.conversions.import_featureclass import *
# from eis_toolkit.conversions.export_featureclass import *
# from eis_toolkit.conversions.export_grid import *
from import_featureclass import *

from bernd.randomforest_classifier_alt import *
from bernd.mlp_classifier_alt import *
from bernd.binarization_inImport_alt import *
from nodata_replace import *
from separation import *
from randomforest_regressor import *
from bernd.logistic_regression_alt import *
from bernd.mlp_regressor_alt import *
from bernd.model_prediction import *
from model_fit import *
from model_testsplit import *
from onehotencoder import *
from unification import *
import fiona
import joblib
from featureclass_to_geopandas import *
from sklearn.preprocessing import OneHotEncoder

# scripts = r'/eis_toolkit'  #/eis_toolkit/conversions'
# sys.path.append (scripts)

# import fiona
# help(fiona.open)
parent_dir = Path(__file__).parent
name_fc = parent_dir.joinpath(r'data/shps/IOCG_Deps_Prosp_Occs.shp') 
dfg = gpd.read_file(name_fc)

urdf = copy.deepcopy(dfg)

fields=  {'OBJECTID':'i', 'ID':'n', 'Name':'n', 'Alternativ':'n', 'Easting_EU':'n', 'Northing_E':'n',
       'Easting_YK':'n', 'Northing_Y':'n', 'Commodity':'c', 'Size_by_co':'c', 'All_main_c':'n',
       'Other_comm':'n', 'Ocurrence_':'n', 'Mine_statu':'n', 'Discovery_':'n', 'Commodity_':'n',
       'Resources_':'n', 'Reserves_t':'n', 'Calc_metho':'n', 'Calc_year':'n', 'Current_ho':'n',
       'Province':'n', 'District':'n', 'Deposit_ty':'n', 'Host_rocks':'n', 'Wall_rocks':'n',
       'Deposit_sh':'c', 'Dip_azim':'v', 'Dip':'t', 'Plunge_azi':'v', 'Plunge_dip':'v', 'Length_m':'v',
       'Width_m':'n', 'Thickness_':'v', 'Depth_m':'n', 'Date_updat':'n', 'geometry':'g'}

df_new,columns= import_featureclass(fc = dfg, fields=fields)
print('****************')
print(df_new)
print('****************')
print(columns)
print('****************')

# nodata deplacement oder _remove

Xvdf, Xcdf, ydf = separation(Xdf = df_new, fields = columns) # t, v, c 

# parent_dir = Path(__file__).parent
# name_enc = parent_dir.joinpath(r'data/myENC.enc') 
# enc1 = joblib.load(name_enc)

# #Umordnen: 
# ydf = Xdf['Dip_azim']       #ydf = Xdf.loc[:,name]
# Xdf.drop('Dip_azim', axis=1, inplace=True)
# Xdf['Dip_azim'] = ydf

# Xdf.drop(Xdf[Xdf.Dip_azim == 270].index, inplace=True)
# tmpb = enc1.transform(Xdf)
# tmpb = pd.DataFrame(tmpb, columns = enc1.get_feature_names_out())  #([].append(col)))


Xdfneu1, enc1 = onehotencoder(Xdf = Xcdf, fields = columns)

Xdf_c,t = nodata_replace(Xdf = Xdfneu1, rtype = 'replace')
Xdf_v,t = nodata_replace(Xdf = Xvdf, rtype = 'replace')
ydf,t = nodata_replace(Xdf = ydf, rtype = 'replace')

Xdf = unification(Xvdf = Xdf_v, Xcdf = Xdf_c)

#Xdf, ydf, nodatmask = nodata_remove(xdf = Xdfneu1,ydf = ydf, target = Ziel)
# ydf aus Xdf lösen
#Xdf, ydf = target_separation (Xdf,columns)

#myML = logistic_regression (Xdf = Xdf, fields = columns)         #ydf = Ziel)
#myML = mlp_regressor (Xdf = Xdf, fields = columns ,hidden_layer_sizes= (100,))              #ydf = Ziel
#myML = mlp_classifier (Xdf = Xdf, fields = columns ,hidden_layer_sizes= (100,))              # ydf = Ziel
#myML = randomforest_classifier (Xdf = Xdf, fields = columns) #ydf = ydf)
myML = randomforest_regressor () #ydf = ydf)
#print('*****************featurenames:')
#print(myML.feature_names_in_)
#print('*****************params:')
#print(myML.get_params())

# ############################### - hier dazwischen kann export und import von myML, columns und cats liegen 
# urdf = copy.deepcopy(Xdf)

# validation = model_testsplit (myML = myML, Xdf = Xdf, ydf = ydf, fields = columns, test_size = 0.2)

myML = model_fit (myML = myML, Xdf = Xdf, ydf = ydf, fields = columns)

parent_dir = Path(__file__).parent
name_mdl = parent_dir.joinpath(r'data/myML.mdl') 
joblib.dump(myML, name_mdl)

# Xdf, ydf = target_separation (Xdf,columns)
#Xdf, ziel ,nodatmask = nodata_remove(df = Xdf2)
# Xdf = nodata_replace(df = Xdf2, rtype = 'k_neighbors', n_neighbors=1)

#validation = model_testsplit(myML, Xdf, )


# t1 = Xdf['Dip_azim_0']       #ydf = Xdf.loc[:,name]
# Xdf.drop('Dip_azim_0', axis = 1, inplace=True)
# Xdf['Dip_azim_0'] = t1

# Xdf.drop(Xdf[Xdf.Dip_azim == 270].index, inplace=True)

parent_dir = Path(__file__).parent
name_mdl = parent_dir.joinpath(r'data/myML.mdl') 
myML1 = joblib.load(name_mdl)


ydf = model_prediction(myML = myML1,Xdf = Xdf)
#print('****************ydf')
#print(ydf)

# #################### ydf an XDF anfügen

#Ergebnis = export_featurclass(Xdf = Xdf, df = ydf, nanmask = nodatmask)
Ergebnis = export_featureclass(Xdfg = urdf, ydf = ydf) #, df = ydf, nanmask = nodatmask)

parent_dir = Path(__file__).parent
name_fc = parent_dir.joinpath(r'data/Ergebnis.shp') 
Ergebnis.to_file(name_fc)
