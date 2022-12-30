"""
Created an November 24 2022
@author: torchala 
""" 
########################
# Test frame for single modules
# 

from pathlib import Path
import copy
import sys
from typing import Tuple
import numpy as np
import rasterio
import os
import matplotlib.pyplot as plt
from rasterio.plot import show
#from PIL import Image

from rasterio.enums import Resampling
# parent_dir = Path(__file__).parent.parent
# scripts = parent_dir.joinpath(r'eis_toolkit/converions')
scripts = r'/eis_toolkit'  #/eis_toolkit/conversions'
sys.path.append (scripts)
from eis_toolkit.conversions.raster_to_pandas import *

from bernd.import_grid import *
from randomforest_regressor import *
from bernd.randomforest_classifier import *
from bernd.mlp_regressor import *
from bernd.mlp_classifier import *
from bernd.model_prediction import *
from export_grid import *
from nodata_replace import *
from nodata_remove import *
from bernd.logistic_regression import *
from model_fit import *
from model_testsplit import *
import datetime
import joblib

from export_featureclass import *
from import_featureclass import *
from separation import *
from bernd.model_prediction import *
from model_fit import *
from model_testsplit import *
from onehotencoder import *
from unification import *
from featureclass_to_geopandas import *
from sklearn.preprocessing import OneHotEncoder

dt = datetime.datetime.now()
print ('*****************************************************')
print('\nStart of frame_image: ' + dt.strftime('%d.%h.%y; %H:%M:%S'))
# Aktuell Testrahmen des Einlesens eines GRID 
# Dateiname eines GRID: 
parent_dir = Path(__file__).parent
print('****************start')
#name_tif = parent_dir.joinpath(r'data/remote/small_raster_16.tif') 
# name_tif1 = parent_dir.joinpath(r'data/Primary_data/AEM/IOCG_AEM_Inph_.tif')
# name_tif2 = parent_dir.joinpath(r'data/Primary_data/AEM/IOCG_AEM_Quad.tif')
# name_tif3 = parent_dir.joinpath(r'data/Primary_data/AEM/IOCG_EM_ratio.tif')
# name_tif4 = parent_dir.joinpath(r'data/Primary_data/Mag/IOCG_Mag_grysc_DGRF65_anom_.tif')
# name_tif5 = parent_dir.joinpath(r'data/Primary_data/Rad/IOCG_Gm_Rd_K_.tif')
# name_tif6 = parent_dir.joinpath(r'data/Primary_data/Rad/IOCG_Gm_Rd_Th_eq_.tif')
# name_tif7 = parent_dir.joinpath(r'data/Primary_data/Rad/IOCG_Gm_Rd_Total_Count_.tif')
# name_tif8 = parent_dir.joinpath(r'data/Primary_data/Rad/IOCG_Gm_Rd_U_eq_.tif')
# target_tif = parent_dir.joinpath(r'data/Primary_data/AEM/IOCG_App_res.tif')

target_tif = parent_dir.joinpath(r'data/test2k.tif') 
# image = Image.open(name_tif)
# image.show()
#name_tif = r'D:\Projekte\_EIS\Daten\EIS_data\EIS_IOCG_target_area_CLB\Primary_data\Mag\IOCG_Mag_grysc_DGRF65_anom_.tif'
#raster = rasterio.open(name_tif)
# df= raster_to_pandas(raster = raster) #, bands = None, add_img_coord = True)

#Ziel,metadata = import_grid(grid = target_tif, name = 'Target') #, add_img_coord = True)
#Xdf1,metadata = import_grid(grid = name_tif, name = 'IOCG_AEM_Inph') #, add_img_coord = True)
#Xdf0,metadata = import_grid(grid = name_tif1, name = 'AEM_Inph_')
##Xdf0 = copy.deepcopy(Ziel)
print('****************')
#print (Xdf0.to_numpy())
name_tif = parent_dir.joinpath(r'data/test1k.tif') 
#name_tif = parent_dir.joinpath(r'data/Primary_data/AEM/IOCG_AEM_Quad.tif')
print (name_tif)

# grids=  [{'name':'test1k', 'file':name_tif, 'type':'v'},
# {'name':'test2k', 'file':target_tif, 'type':'c'},
# {'name':'testt', 'file':target_tif, 'type':'t'}]

#name_tif = parent_dir.joinpath(r'data/remote/small_raster_16.tif') 
name_tif1 = parent_dir.joinpath(r'data/Primary_data/AEM/IOCG_AEM_Inph_.tif')
name_tif2 = parent_dir.joinpath(r'data/Primary_data/AEM/IOCG_AEM_Quad.tif')
name_tif3 = parent_dir.joinpath(r'data/Primary_data/AEM/IOCG_EM_ratio.tif')
name_tif4 = parent_dir.joinpath(r'data/Primary_data/Mag/IOCG_Mag_grysc_DGRF65_anom_.tif')
name_tif5 = parent_dir.joinpath(r'data/Primary_data/Rad/IOCG_Gm_Rd_K_.tif')
name_tif6 = parent_dir.joinpath(r'data/Primary_data/Rad/IOCG_Gm_Rd_Th_eq_.tif')
name_tif7 = parent_dir.joinpath(r'data/Primary_data/Rad/IOCG_Gm_Rd_Total_Count_.tif')
name_tif8 = parent_dir.joinpath(r'data/Primary_data/Rad/IOCG_Gm_Rd_U_eq_.tif')
target_tif = parent_dir.joinpath(r'data/Primary_data/AEM/IOCG_App_res.tif')

grids=  [{'name':'AEM_Inph', 'file':name_tif1, 'type':'v'},
{'name':'AEM_Quad', 'file':name_tif2, 'type':'v'},
{'name':'EM_ratio', 'file':name_tif3, 'type':'v'},
{'name':'App_res', 'file':target_tif, 'type':'t'}]

#Xdf2 = copy.deepcopy(Xdf1)
print('****************')
print(grids)

Xdf,fields, metadata = import_grid(grids)

Xdf,nanm = nodata_remove(Xdf = Xdf)

Xvdf, Xcdf, ydf = separation(Xdf = Xdf, fields = fields)

Xdfneu1, ohe = onehotencoder(Xdf = Xcdf, fields = fields)
# Xdfneu1, ohe = onehotencoder(Xdf = Xcdf, ohe = ohe)   prediction

Xdf = unification(Xvdf = Xvdf, Xcdf = Xdfneu1)

dt = datetime.datetime.now()
print ('*****************************************************')
print('\nEnde Import und nodataremove: ' + dt.strftime('%d.%h.%y; %H:%M:%S'))

#myML = logistic_regression (Xdf = Xdf1, ydf = Ziel)
#myML = mlp_regressor (Xdf = Xdf1, ydf = Ziel ,hidden_layer_sizes= (100,))
#myML = mlp_classifier (Xdf = Xdf, ydf = ydf ,hidden_layer_sizes= (100,))
myML = randomforest_regressor () #oob_score = True)
#myML = randomforest_classifier (Xdf = Xdf, ydf = ydf)
#print('*****************featurenames:')
#print(myML.feature_names_in_)
#print('*****************params:')
#print(myML.get_params())

print('*****************testsplit Start:')
print ('*****************************************************')
print('\nStart Testsplit: ' + dt.strftime('%d.%h.%y; %H:%M:%S'))
validation = model_testsplit (myML = myML, Xdf = Xdf, ydf = ydf, test_size = 0.2)
print ('*****************************************************')
print('\nEnde Testsplit: ' + dt.strftime('%d.%h.%y; %H:%M:%S'))
print('*****************validation:')
print(validation)

myML = model_fit (myML = myML, Xdf = Xdf, ydf = ydf)
dt = datetime.datetime.now()
print ('*****************************************************')
print('\nEnde RF Regressor Training: ' + dt.strftime('%d.%h.%y; %H:%M:%S'))

parent_dir = Path(__file__).parent
name_mdl = parent_dir.joinpath(r'data/myML.mdl') 
name_metadata = parent_dir.joinpath(r'data/myMeta.mta') 
name_fields = parent_dir.joinpath(r'data/myFields.fld') 
name_ohe = parent_dir.joinpath(r'data/myOhe.ohe') 

joblib.dump(myML, name_mdl)
joblib.dump(metadata, name_metadata)
joblib.dump(fields, name_fields)
joblib.dump(ohe, name_ohe)

##### Prediction

# myML1 = joblib.load(name_mdl)
# myMetadata = joblib.load(name_metadata)
# myFields = joblib.load(name_fields)

# # Noch: PrÜfung fields (reihenfolge) mit fields und myML1 
 
# ydf = model_prediction(myML=myML1,Xdf=Xdf)
# print('****************ydf')
# print(ydf)

# dt = datetime.datetime.now()
# print ('*****************************************************')
# print('\nEnde RF Prediction: ' + dt.strftime('%d.%h.%y; %H:%M:%S'))

# #Ergebnis = export_grid(metadata = metadata, df = ydf, nanmask = nodatmask)
# Ergebnis = export_grid(metadata = myMetadata, df = ydf, nanmask = nanm)
# #print('****************Ergebnis')
# #print(Ergebnis)
# #print('ok ###########################')

# profile={
# 'count': 1,  'driver': 'GTiff',
# 'dtype': 'float32',  
# 'tiled': False,
# 'width':9, # 100,
# 'height':5  #200
# }

# ### Tiff ausgeben: (pandas_to_raster)
# parent_dir = Path(__file__).parent
# outpath = parent_dir.joinpath(r'data')   # /test1.tif')
# print (outpath)

# with rasterio.open(os.path.join(outpath, 'ergebnis.tif'), 'w', **profile) as dst:
#     dst.write_band(1, Ergebnis.astype(rasterio.float64))

