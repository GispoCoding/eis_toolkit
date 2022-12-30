"""
Test split validation for a ML-model
Created an Dezember 10 2022
@author: torchala 
""" 
#### Stand:  fast fertig 
#        - Tests 

from typing import Optional, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from model_prediction import *
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************

def _model_testsplit(  # type: ignore[no-any-unimported]
   myML: Any,
   Xdf: pd.DataFrame,      # dataframe of Features for traning
   ydf: Optional[pd.DataFrame] = None,      # dataframe of known values for training
   fields: Optional[dict] = None,
   test_size: Optional[int | float] = None,  # int: namuber of test-samples, if float: 0<ts<1
   train_size: Optional[int | float] = None,  # if None: complement size of the test_size
   random_state: Optional [int] = None,
   shuffle: Optional [bool] = None
   #stratify: Optional [np.array] = None, 
) -> dict:

   # if ydf not as an separated datafram: separat "t"-column out of Xdf
   if ydf is None:
      if fields is None:
         raise InvalidParameterValueException ('***  target and target-field are None: ') 
      else:
         name = {i for i in fields if fields[i]=="t"}
         ydf = Xdf[list(name)]
         Xdf.drop(name, axis=1, inplace=True)

   # split in test and training datasets
   train_X, test_X, train_y, test_y = train_test_split(
      Xdf,
      ydf,
      test_size = test_size,
      train_size = train_size,
      random_state = random_state,
      shuffle = shuffle)
   
   ty = train_y
   if len(train_y.shape) > 1:
      if train_y.shape[1] == 1:
          ty = np.ravel(train_y)

   #  Training
   myML.fit(train_X, ty)

   # Validation  
   validation ={}
   if hasattr(myML,'oob_score_'):
         validation['oob'] = myML.oob_score_
   predict_y = model_prediction(myML,test_X)
   if myML._estimator_type == 'regressor':
      validation["R2"] = metrics.r2_score(test_y,predict_y)
      validation["explained_variance"] = metrics.explained_variance_score(test_y,predict_y)
      validation["mean"] = metrics.mean_absolute_error(test_y,predict_y)
      validation["mse"] = metrics.mean_squared_error(test_y,predict_y)
   else:
      validation["curacy"] = metrics.accuracy_score(test_y,predict_y)
      validation["recall"] = metrics.recall_score(test_y,predict_y)
      validation["precision"] = metrics.precision_score(test_y,predict_y)
      validation["F1"] = metrics.f1_score(test_y,predict_y)

   return validation

# *******************************
def model_testsplit(  # type: ignore[no-any-unimported]
   myML, 
   Xdf: pd.DataFrame,      # dataframe of Features for traning
   ydf: Optional[pd.DataFrame] = None,      # dataframe of known values for training
   fields: Optional[dict] = None,
   test_size: Optional[int | float] = None,  # int: namuber of test-samples, if float: 0<ts<1
   train_size: Optional[int | float] = None,  # if None: complement size of the test_size
   random_state: Optional [int] = None,
   shuffle: Optional [bool] = None
   #stratify: Optional [np.array] = None
) -> dict:

   """ testsplit validation for a ML model

   Args:
   - Xdf: Pandas dataframe or numpy array ("array-like") of features (columns) and samples (rows)
   - ydf: Pandas dataframe or numpy array ("array-like") of target valus(columns) and samples (rows) (same number as Xdf)
        If ydf is = None, target column is included i Xdf. In this case fields should not be None
   - fields: dictionary of the fieldnames and type of fields. A field type 't' is needed, fields is not needed if ydf is not None.
   - test_size: float or int, default=None 
      If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. 
      If int, represents the absolute number of test samples. 
      If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.
   - train_size: float or int, default=None
      If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. 
      If int, represents the absolute number of train samples. 
      If None, the value is automatically set to the complement of the test size.
   - random_state: int, RandomState instance or None, default=None
      Controls the shuffling applied to the data before applying the split. 
      Pass an int for reproducible output across multiple function calls.
   - shuffle bool, default=True
      Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.
   - stratify array-like, default=None
      If not None, data is split in a stratified fashion, using this as the class labels.
    
   Returns:
        fited ML model
   """

   validation = _model_testsplit(
      myML = myML,
      Xdf = Xdf, 
      ydf = ydf, 
      fields = fields,
      test_size = test_size,
      train_size = train_size,
      random_state = random_state,
      shuffle = shuffle 
   )

   return validation

