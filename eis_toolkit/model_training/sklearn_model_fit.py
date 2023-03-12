
from typing import Any
import numpy as np
import pandas as pd
from eis_toolkit.exceptions import InvalidParameterValueException, InvalideContentOfInputDataFrame

# *******************************

def _sklearn_model_fit(  # type: ignore[no-any-unimported]
   sklearnMl: Any,
   Xdf: pd.DataFrame,                       # dataframe of Features for traning
   ydf: pd.DataFrame,      # dataframe of known values for training
):

   ty = ydf
   if len(ydf.shape) > 1: 
        if ydf.shape[1] == 1:
              ty = np.ravel(ydf)

   if sklearnMl._estimator_type == 'classifier':
      if np.issubdtype(ty.dtype, np.floating):
         raise InvalideContentOfInputDataFrame('A classifier model cannot use a float y (target)')
         #ty = (ty + 0.5).astype(np.uint16)
   else:
      if not np.issubdtype(ty.dtype, np.number):
         raise InvalideContentOfInputDataFrame('A regressor model can only use number y (target)')
      
   sklearnMl.fit(Xdf, ty)

   return sklearnMl

# *******************************
def sklearn_model_fit(  # type: ignore[no-any-unimported]
   sklearnMl: Any,
   Xdf: pd.DataFrame,                       # dataframe of Features for training
   ydf: pd.DataFrame,                     # dataframe of known values for training
):

   """ 
      Training of a ML model
   Args:
      - sklearnMl: before defined model (random rorest  classifier, random forest regressor, logistic regressor)
      - Xdf (Pandas dataframe or numpy array ("array-like")): features (columns) and samples (rows)
      - ydf (Pandas dataframe or numpy array ("array-like")): target valus(columns) and samples (rows) (same number as Xdf)
         If ydf is float and the estimator is a classifier: ydf will be rounded to int.
         Returns:
        Fited ML model
   """

   # Argument evaluation
   fl = []
   t = sklearnMl.__class__.__name__           #t = isinstance(sklearnMl,(RandomForestClassifier,RandomForestRegressor,LogisticRegression))
   if not t in ('RandomForestClassifier','RandomForestRegressor','LogisticRegression'):
      fl.append('argument sklearnMl is not an instance of one of (RandomForestClassifier,RandomForestRegressor,LogisticRegression)')
   if not (isinstance(Xdf, pd.DataFrame)):
      fl.append('argument Xdf is not a DataFrame')
   if not (isinstance(ydf, pd.DataFrame)):
      fl.append('argument ydf is not a DataFrame')
   if len(fl) > 0:
      raise InvalidParameterValueException (fl[0])

   fl = []
   if len(Xdf.columns) == 0:
      fl.append('DataFrame Xdf has no column')
   if len(Xdf.index) == 0:
      fl.append('DataFrame Xdf has no rows')
   if len(ydf.columns) != 1:
      fl.append('DataFrame ydf has 0 or more then columns')
   if len(ydf.index) == 0:
      fl.append('DataFrame ydf has no rows')

   if Xdf.isna().sum().sum() > 0 or ydf.isna().sum().sum() > 0:
      fl.append('DataFrame ydf or Xdf contains Nodata-values')
   if len(fl) > 0:
      raise InvalidParameterValueException (fl[0])

   sklearnMl = _sklearn_model_fit(
      sklearnMl = sklearnMl,
      Xdf = Xdf,
      ydf = ydf
   )

   return sklearnMl
