
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from eis_toolkit.prediction_methods.sklearn_model_prediction import *
from sklearn import metrics
from eis_toolkit.exceptions import InvalidParameterValueException, InvalideContentOfInputDataFrame

# *******************************

def _sklearn_model_validations(
   sklearnMl,                            # Optional[Any] = None,   # None: just compare ydf and predict_ydf
   Xdf: Optional[pd.DataFrame] = None,    # dataframe of Features for traning (to split in training and test dataset)
   ydf: Optional[pd.DataFrame] = None,    # dataframe of known values for training (to split) or known values to compare with test_y
   predict_ydf: Optional[pd.DataFrame] = None,   # predicted values to compare with ydf (known values), if given Xdf is not nessesarry
   test_size: Optional[int | float] = None,  # int: number of test-samples, if float: 0<ts<1
   train_size: Optional[int | float] = None, # if None: complement size of the test_size
   random_state: Optional [int] = None,
   shuffle: Optional [bool] = True,
   confusion_matrix: Optional[bool] = True,
   comparison: Optional[bool] = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:

   # configuration:
   maxlines = 1.000    # max. number of y ->rows for comparison

   # external tet_set for y will be used
   if predict_ydf is not None:
      if ydf.shape[0] != predict_ydf.shape[0]:
         raise InvalidParameterValueException ('Known testset (y) and predicted y have not the same number of samples')
      test_y = ydf
      predict_y = predict_ydf
      testtype = "test_dataset"
   
   else:    # split in test and training datasets
      testtype = "test_split"
      if (test_size is None) and (train_size is None):
         test_size = 0.25
      if test_size is not None:
         if test_size != 0:           # if testsize == 0 :   selftest will be performed
            train_X, test_X, train_y, test_y = train_test_split(
               Xdf,
               ydf,
               test_size = test_size,
               random_state = random_state,
               shuffle = shuffle)
         else:
            test_y = train_y = ydf
            test_X = train_X = Xdf
            testtype = 'self_test'
      elif train_size is not None:
         if train_size != 0:   # if trainsize == 0:   selftest will be performed
            train_X, test_X, train_y, test_y = train_test_split(
               Xdf,
               ydf,
               train_size = train_size,
               random_state = random_state,
               shuffle = shuffle)
         else:
            test_y = train_y = ydf
            test_X = train_X = Xdf
            testtype = 'self_test'

      #  Training based on the training-data
      ty = train_y
      if len(train_y.shape) > 1: 
         if train_y.shape[1] == 1:
               ty = np.ravel(train_y)

      if sklearnMl._estimator_type == 'classifier':
         if np.issubdtype(ty.dtype, np.floating):
            raise InvalideContentOfInputDataFrame('A classifier model cannot us a float y (target)')
            #ty = (ty + 0.5).astype(np.uint16)
      else:
         if not np.issubdtype(ty.dtype, np.number):
            raise InvalideContentOfInputDataFrame('A regressor model can only use number y (target)')
   
      if train_X.isna().sum().sum() > 0 or train_y.isna().sum().sum() > 0:
         raise InvalideContentOfInputDataFrame('DataFrame ydf or Xdf contains Nodata-values')
      
      sklearnMl.fit(train_X,ty)

      #Prediction based on the test-data
      if test_X.isna().sum().sum() > 0:
         raise InvalideContentOfInputDataFrame('DataFrame Xdf contains Nodata-values')
      
      predict_y = sklearn_model_prediction(sklearnMl,test_X)

   # Validation 
   validation ={}
   if sklearnMl._estimator_type == 'regressor':
      validation["R2 score"] = metrics.r2_score(test_y, predict_y)
      validation["explained variance"] = metrics.explained_variance_score(test_y, predict_y)
      validation["mean absolut error"] = metrics.mean_absolute_error(test_y, predict_y)
      validation["mean square arror"] = metrics.mean_squared_error(test_y, predict_y)
   else:
      validation["accuracy"] = metrics.accuracy_score(test_y, predict_y)
      validation["recall"] = metrics.recall_score(test_y, predict_y, average = 'weighted') #'macro')
      validation["precision"] = metrics.precision_score(test_y, predict_y,average = 'weighted') #'macro')
      validation["F1 score"] = metrics.f1_score(test_y, predict_y, average = 'weighted') #'macro')
   #if sklearnMl.estimator_ in ['RandomForestClassifier','RandomForesteRegressor']:
   if hasattr(sklearnMl,'oob_score'):
      if sklearnMl.oob_score:
         validation['oob score'] = sklearnMl.oob_score_
   validation['testsplit size'] = test_y.shape[0]
   validation =  pd.DataFrame.from_dict(validation, orient='index', columns=[testtype]) 

   # confusion matrix
   confusion1 = None
   if sklearnMl._estimator_type == 'classifier':
      if confusion_matrix:
         ltest = test_y.loc[:,test_y.columns[0]].tolist()
         lpredict = predict_y.loc[:,predict_y.columns[0]].tolist()
         lists = list(set(ltest+lpredict))
         lists.sort()
         confusion = pd.DataFrame(metrics.confusion_matrix(ltest, lpredict))  #,labels=lists))      #,predict_y, labels=list1))
         list2 = list(confusion.index.values)
         df1 = confusion.rename(index=dict(zip(list2,lists)))
         confusion1 = df1.rename(columns=dict(zip(list2,lists)))
   # sklearnMl.classes_  labels in sklearnMl n_classes_ (number of different label )

   #comparison
   comparison_lst = None
   if comparison and test_y.shape[0] <= maxlines:
      # if test_Id is not None:   #(with ID-Column??)
      #       tmpl = pandas.DataFrame(test_Id, columns = ['Id']).join(test_y).join(test_pred)
      #   else:
      #tmpl = test_y.column[0].join(predict_y.columns[0])
      predict_y.reset_index(drop=True, inplace = True)
      test_y.reset_index(drop=True, inplace = True)
      comparison_lst = test_y.join(predict_y)

   return validation, confusion1, comparison_lst, sklearnMl

# *******************************
def sklearn_model_validations(
   sklearnMl,
   Xdf: Optional[pd.DataFrame] = None,      # dataframe of Features for splitting in traning and test dataset
   ydf: Optional[pd.DataFrame] = None,      # dataframe of known values for splitting 
   predict_ydf: Optional[pd.DataFrame] = None,   # if No random subset will be used for validation (test_size...)
   test_size: Optional[int | float] = None,  # int: namuber of test-samples, if float: 0<ts<1
   train_size: Optional[int | float] = None,  # if None: complement size of the test_size
   random_state: Optional [int] = None,
   shuffle: Optional [bool] = True,
   confusion_matrix: Optional[bool] = True,   # calculate confusion matrix
   comparison: Optional[bool] = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:  #dict, dict]:

   """ 
      Validation for a ML model based on:
      - Randomly splited testset from size test_size/train_size. 
        Xdf and ydf will be randomly splitted in a test and a training dataset. 
        The training-dataset will be used for model-training. 
        The test-dataset will be used for prediction. 
        The result of prediction will be compared with ydf from test-dataset
      - If predict_ydf ist given:  
        ydf is the known set of data to compare with the predicted set predict_ydf.
        sklearnMl will be used to determine the estimator type: regression or classification.
   Args:
      - sklearnMl (model): Even for comparison with a testset the model is used to get the model-typ (regression or classification). 
        If ydf and predict_ydf will be compared. sklearnMl has te information whether ist a classification or a regression-model
      - Xdf Pandas dataframe or numpy array ("array-like"): Features (columns) and samples (rows)
      - ydf Pandas dataframe or numpy array ("array-like"): Target valus(column) and samples (rows) (same number as Xdf).
        If ydf is float and the estimator is a classifier: ydf will be rounded to int.
      - predict_ydf:  Pandas dataframe or numpy array ("array-like"): Predicted values of a test_dataset, to compare with ydf
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
         No effect with selftest.
      - shuffle (bool, default=True):
         Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.
         No effect with selftest
      - confusion_matrix (bool, default = True): If True a confusion matrix will be generated if the model is a classifiing estimator.
      _ comparison (bool, default = False): If True and there are less then 1.000 rows in X.
   
   Returns:
      - validation: DataFrame with all values of the validation
      - confusion_matrix: DataFrame of confusion matrix, if calculated.
           confusion should be read paire wise: 
           number of pair test(0),predict(0), 
           up to n classes which apeers in y_test and y_predict.
      - comparison_list: pd.DataFrame if calculated.
      - model, if calculated
   """

   # Argument evaluation
   fl = []
   t = sklearnMl.__class__.__name__           #t = isinstance(sklearnMl,(RandomForestClassifier,RandomForestRegressor,LogisticRegression))
   if not t in ('RandomForestClassifier','RandomForestRegressor','LogisticRegression'):
      fl.append('Argument sklearnMl is not an instance of one of (RandomForestClassifier,RandomForestRegressor,LogisticRegression)')
   if not (isinstance(Xdf, pd.DataFrame) or (Xdf is None)):
      fl.append('Argument Xdf is not a DataFrame and is not None')
   if not (isinstance(ydf, pd.DataFrame)  or (ydf is None)):
      fl.append('Argument ydf is not a DataFrame and is not None')
   if not (isinstance(predict_ydf, pd.DataFrame) or (predict_ydf is None)):
      fl.append('Argument predict_ydf is not a DataFrame and is not None')

   if not (isinstance(test_size, (float,int)) or (test_size is None)):
      fl.append('Argument test_size is not integer and is not None')
   if not (isinstance(train_size, (float,int)) or (train_size is None)):
      fl.append('Argument train_size is not integer and is not None')
   if not (isinstance(random_state, int) or (random_state is None)):
      fl.append('Argument random_state is not integer and is not None')
   if not (isinstance(shuffle, bool) or (shuffle is None)):
      fl.append('Argument shuffle is not bool and is not None')
   if not (isinstance(confusion_matrix, bool) or (confusion_matrix is None)):
      fl.append('Argument confusion_matrix is not bool and is not None')
   if not (isinstance(comparison, bool) or (comparison is None)):
      fl.append('Argument comparison is not bool and is not None')
   if len(fl) > 0:
      raise InvalidParameterValueException(fl[0])

   if Xdf is not None:
      if len(Xdf.columns) == 0:
         raise InvalidParameterValueException('DataFrame Xdf has no column')
      if len(Xdf.index) == 0:
         raise InvalidParameterValueException('DataFrame Xdf has no rows')
   if predict_ydf is not None:
      if len(predict_ydf.columns) != 1:
        raise InvalidParameterValueException('predict_ydf has 0 or more than 1 columns')
   if predict_ydf is not None:
      if ydf is None:
         raise InvalidParameterValueException('DataFrame ydf is None')
      elif predict_ydf.shape[0] != ydf.shape[0]:
         raise InvalideContentOfInputDataFrame('predict_ydf and ydf have not the same number of rows')
   if ydf is not None: 
      if len(ydf.columns) != 1:
         raise InvalidParameterValueException('DataFrame ydf has 0 or more than 1 columns')
      if len(ydf.index) == 0:
         raise InvalidParameterValueException('DataFrame ydf has no rows')     

   validation, confusion, comparison, sklearnMl = _sklearn_model_validations(
      sklearnMl = sklearnMl,
      Xdf = Xdf,
      ydf = ydf,
      predict_ydf = predict_ydf,
      test_size = test_size,
      train_size = train_size,
      random_state = random_state,
      shuffle = shuffle,
      confusion_matrix = confusion_matrix,
      comparison = comparison,
   )

   return validation, confusion, comparison, sklearnMl
