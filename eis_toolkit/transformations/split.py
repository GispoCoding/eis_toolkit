
from typing import Optional,Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from eis_toolkit.exceptions import InvalidParameterValueException, InvalideContentOfInputDataFrame

# *******************************

def _split(  # type: ignore[no-any-unimported]
   Xdf: pd.DataFrame,    # dataframe of Features for traning (to split in training and test dataset) and or for Xdf abnd ydf
   ydf: Optional[pd.DataFrame] = None,    # dataframe of known values for training (to split) or known values to compare with test_y
   fields: Optional[dict] = None,        # 't'-field will be used if ydf is None 
   test_size: Optional[int | float] = None,  # int: number of test-samples, if float: 0<ts<1
   train_size: Optional[int | float] = None, # if None: complement size of the test_size
   random_state: Optional [int] = None,
   shuffle: Optional [bool] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:


   # if ydf not as an separated datafram: separat "t"-column out of Xdf
   if ydf is None:
      if fields is not None:
         name = {i for i in fields if fields[i]=="t"}
         ydf = Xdf[list(name)]
         Xdf.drop(name,axis=1,inplace=True)

   # check:
   if test_size is not None:
      if test_size <= 0:
         test_size = None
   if train_size is not None:         
      if train_size <= 0:
         train_size = None   
   if test_size is None and train_size is None:
      train_Xdf = Xdf
      if ydf is None:
         train_ydf = None
      else:
         train_ydf = ydf
   else:
      # split in test and training datasets
      if ydf is not None:
         train_Xdf,test_Xdf,train_ydf,test_ydf = train_test_split(
         Xdf,
         ydf,
         test_size = test_size,
         train_size = train_size,
         random_state = random_state,
         shuffle = shuffle)
      else:
         train_ydf = test_ydf = None
         train_Xdf,test_Xdf = train_test_split(
            Xdf,
            test_size = test_size,
            train_size = train_size,
            random_state = random_state,
            shuffle = shuffle)
      # train_Xdf = pd.Dataframe(train_X,columns=Xdf.columns)
      # test_Xdf = pd.Dataframe(test_X,columns=Xdf.columns)
      # train_xdf = pd.Dataframe(train_y,columns=ydf.columns)
      # test_ydf = pd.Dataframe(test_y,columns=Xdf.columns)
   return train_Xdf, test_Xdf, train_ydf, test_ydf

# *******************************
def split(  # type: ignore[no-any-unimported]
   Xdf: pd.DataFrame,      # dataframe of Features for splitting in traning and test dataset
   ydf: Optional[pd.DataFrame] = None,      # dataframe of known values for splitting 
   fields: Optional[dict] = None,
   test_size: Optional[int | float] = None,  # int: namuber of test-samples, if float: 0<ts<1
   train_size: Optional[int | float] = None,  # if None: complement size of the test_size
   random_state: Optional [int] = None,
   shuffle: Optional [bool] = None,
) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:

   """ 
      Splits Xdf and ydf randomly in training set and in test set.
      If ydf is not given and fields is None: Xdf will be splitted.
      If ydf is None and fields include a target field (type = 't') ydf will be created out of Xdf.
   Args:
      - Xdf Pandas dataframe or numpy array ("array-like"): Features (columns) and samples (rows)
      - ydf Pandas dataframe or numpy array ("array-like"): target valus(columns) and samples (rows) (same number as Xdf)
         If ydf is = None, target column is included in Xdf. In this case "fields" should not be None. Is fild None, Xdf will be splitted
      - fields (dictionary): shold be given if ydf ist a column inside ydf (fields-type = 't')
      - test_size (float or int, default=None): 
         If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. 
         If int, represents the absolute number of test samples. 
         If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.
      - train_size (float or int, default=None):
         If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. 
         If int, represents the absolute number of train samples. 
         If None, the value is automatically set to the complement of the test size.
      - random_state (int, RandomState instance or None, default=None):
         Controls the shuffling applied to the data before applying the split. 
         Pass an int for reproducible output across multiple function calls.
         No effect with selftest
      - shuffle (bool, default=True):
         Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.
         No effect with selftest
   Returns:
      Pandas Dataframes: train_X, test_X, (train_y, test_y)
   """

   # Argument evaluation
   fl = []
   if not (isinstance(Xdf ,pd.DataFrame)):
      fl.append('Argument Xdf is not a DataFrame and is not None')
   if not (isinstance(ydf ,pd.DataFrame)  or (ydf is None)):
      fl.append('Argument ydf is not a DataFrame and is not None')
   if not (isinstance(test_size, (float,int)) or (test_size is None)):
      fl.append('Argument test_size is not integer and is not None')
   if not (isinstance(train_size, (float,int)) or (train_size is None)):
      fl.append('Argument train_size is not integer and is not None')
   if not (isinstance(random_state, int) or (random_state is None)):
      fl.append('Argument random_state is not integer and is not None')
   if not (isinstance(shuffle, bool) or (shuffle is None)):
      fl.append('Argument shuffle is not bool and is not None')
   if not ((isinstance(fields, dict)) or fields is None):
      fl.append('Argument fields is not a dictionary and is not None')

   if len(fl) > 0:
      raise InvalidParameterValueException(fl[0])

   if len(Xdf.columns) == 0 or len(Xdf.index) == 0:
      raise InvalideContentOfInputDataFrame('DataFrame has no column or no rows')


   return _split(
      Xdf = Xdf,
      ydf = ydf,
      fields = fields,
      test_size = test_size,
      train_size = train_size,
      random_state = random_state,
      shuffle = shuffle
   )

