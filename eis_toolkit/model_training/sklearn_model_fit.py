
from typing import Any
import numpy as np
import pandas as pd
from eis_toolkit.exceptions import InvalidParameterValueException

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
   if len(ydf.columns) == 0:
      fl.append('DataFrame ydf has no column')
   if len(ydf.index) == 0:
      fl.append('DataFrame ydf has no rows')
   if len(fl) > 0:
      raise InvalidParameterValueException (fl[0])


   sklearnMl = _sklearn_model_fit(
      sklearnMl = sklearnMl,
      Xdf = Xdf,
      ydf = ydf
   )

   return sklearnMl
