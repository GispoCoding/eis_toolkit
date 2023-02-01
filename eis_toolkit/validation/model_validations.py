"""
Test split validation for a ML-model
Created an Dezember 10 2022
@author: torchala 
""" 
## zum Verzeichnis validation
#### Stand:  fast fertig 
#        - Tests 

from typing import Optional, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from model_prediction import *
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************

def _model_validations(  # type: ignore[no-any-unimported]
   myML: Any,
   Xdf: pd.DataFrame,      # dataframe of Features for traning
   ydf: Optional[pd.DataFrame] = None,      # dataframe of known values for training
   fields: Optional[dict] = None,
   test_size: Optional[int | float] = None,  # int: number of test-samples, if float: 0<ts<1
   train_size: Optional[int | float] = None,  # if None: complement size of the test_size
   random_state: Optional [int] = None,
   shuffle: Optional [bool] = None,
   confusion_matrix: Optional[bool] = True
   #stratify: Optional [np.array] = None, 
) -> Tuple[Any, pd.DataFrame, pd.DataFrame]:  #dict

   if len(Xdf.columns) == 0:
      raise InvalidParameterValueException ('***  DataFrame has no column')
   if len(Xdf.index) == 0:
      raise InvalidParameterValueException ('***  DataFrame has no rows')

   # if ydf not as an separated datafram: separat "t"-column out of Xdf
   if ydf is None:
      if fields is None:
         raise InvalidParameterValueException ('***  target and target-field are None: ') 
      else:
         name = {i for i in fields if fields[i]=="t"}
         ydf = Xdf[list(name)]
         Xdf.drop(name,axis=1,inplace=True)

   # split in test and training datasets
   selftest = False
   if test_size is not None:
      if test_size != 0:           # if testsize == 0 :   selftest will be performed
         train_X,test_X,train_y,test_y = train_test_split(
            Xdf,
            ydf,
            test_size = test_size,
            train_size = train_size,
            random_state = random_state,
            shuffle = shuffle)
      else:
         test_y = train_y = ydf
         test_X = train_X = Xdf
         selftest = True
   elif train_size is not None:
            if train_size != 0:   # if trainsize == 0:   selftest will be performed
               train_X,test_X,train_y,test_y = train_test_split(
                  Xdf,
                  ydf,
                  test_size = test_size,
                  train_size = train_size,
                  random_state = random_state,
                  shuffle = shuffle)
            else:
               test_y = train_y = ydf
               test_X = train_X = Xdf
               selftest = True

   # ty = train_y
   # if len(train_y.shape) > 1:
   #    if train_y.shape[1] == 1:
   #        ty = np.ravel(train_y)

   #  Training
   myML.fit(train_X,train_y)

   # Validation 
   validation ={}
   predict_y = model_prediction(myML,test_X)
   
   if myML._estimator_type == 'regressor':
      validation["R2"] = metrics.r2_score(test_y,predict_y)
      validation["explained_variance"] = metrics.explained_variance_score(test_y,predict_y)
      validation["mean"] = metrics.mean_absolute_error(test_y,predict_y)
      validation["mse"] = metrics.mean_squared_error(test_y,predict_y)
   else:
      validation["accuracy"] = metrics.accuracy_score(test_y,predict_y)
      validation["recall"] = metrics.recall_score(test_y,predict_y,average = 'weighted') #'macro')
      validation["precision"] = metrics.precision_score(test_y,predict_y,average = 'weighted') #'macro')
      validation["F1"] = metrics.f1_score(test_y,predict_y,average = 'weighted') #'macro')
   #if myML.estimator_ in ['RandomForestClassifier','RandomForesteRegressor']:
   if hasattr(myML,'oob_score'):
      if myML.oob_score:
         validation['oob_score_'] = myML.oob_score_
   validation['testsplit size'] = test_y.shape[0]
   if selftest:   # Self-Test
      t = ["self_test"]
   else:
      t = ["test_split"]
   validation =  pd.DataFrame.from_dict(validation,orient='index', columns=t) 

   # confusion matrix
   confusion = None
   if myML._estimator_type == 'classifier':
      if confusion_matrix:
         ltest = test_y.loc[:,test_y.columns[0]].tolist()
         lpredict = predict_y.loc[:,predict_y.columns[0]].tolist()
         lists = list(set(ltest+lpredict))
         lists.sort()
         confusion = pd.DataFrame(metrics.confusion_matrix(ltest,lpredict))#,labels=lists))      #,predict_y, labels=list1))
         # Beschriften der Tabelle: 
         #sort_y =  test_y.sort_values(test_y.columns[0])
         # Name of the y-columns
         # l1 = test_y.loc[:,test_y.columns[0]].tolist()
         # list1 = list(set(ltest))
         # list1.sort() 
         list2 = list(confusion.index.values)
         df1 = confusion.rename(index=dict(zip(list2,lists)))
         confusion1 = df1.rename(columns=dict(zip(list2,lists)))
   # myML.classes_  labels in myML n_classes_ (number of different label )

   return myML,validation,confusion1

# *******************************
def model_validations(  # type: ignore[no-any-unimported]
   myML, 
   Xdf: pd.DataFrame,      # dataframe of Features for traning
   ydf: Optional[pd.DataFrame] = None,      # dataframe of known values for training
   fields: Optional[dict] = None,
   test_size: Optional[int | float] = None,  # int: namuber of test-samples, if float: 0<ts<1
   train_size: Optional[int | float] = None,  # if None: complement size of the test_size
   random_state: Optional [int] = None,
   shuffle: Optional [bool] = None,
   confusion_matrix: Optional[bool] = True   # calculate confusion matrix
   #stratify: Optional [np.array] = None
) -> Tuple[Any,pd.DataFrame,pd.DataFrame]:  #dict, dict]:

   """ vaidation for a ML model based on splited testset
   Args:
      - Xdf Pandas dataframe or numpy array ("array-like"): features (columns) and samples (rows)
      - ydf Pandas dataframe or numpy array ("array-like"): target valus(columns) and samples (rows) (same number as Xdf)
         If ydf is = None, target column is included i Xdf. In this case fields should not be None
      - fields (dictionary): the fieldnames and type of fields. A field type 't' is needed, fields is not needed if ydf is not None.
      - test_size (float or int, default=None): 
         If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. 
         If int, represents the absolute number of test samples. 
         If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.
         If = 0 or 0.0: a selftest will be peformed: test-set = train-set
      - train_size (float or int, default=None):
         If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. 
         If int, represents the absolute number of train samples. 
         If None, the value is automatically set to the complement of the test size.
         if = 0 or 0.0 a selftest will be performed: : test-set = train-set
      - random_state (int, RandomState instance or None, default=None):
         Controls the shuffling applied to the data before applying the split. 
         Pass an int for reproducible output across multiple function calls.
         No effect with selftest
      - shuffle (bool, default=True):
         Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.
         No effect with selftest
      - stratify (array-like, default=None):
         If not None, data is split in a stratified fashion, using this as the class labels.
         No effect with selftest
    
   Returns:
        Fitted Model
        Dictionary of the validation
        DataFrame of confusion matrix
        confusion schold be read pairwise: 
         number of pair test(0),predict(0), 
         up to n classes which apeers in y_test and y_predict
   """

   myML,validation,confusion = _model_validations(
      myML = myML,
      Xdf = Xdf, 
      ydf = ydf, 
      fields = fields,
      test_size = test_size,
      train_size = train_size,
      random_state = random_state,
      shuffle = shuffle,
      confusion_matrix = confusion_matrix
   )

   return myML,validation,confusion

