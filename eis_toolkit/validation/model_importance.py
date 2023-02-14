"""
Test split validation for a ML-model
Created an Februar 07/23
@author: torchala 
""" 

from typing import Optional, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************

def _model_importance(  # type: ignore[no-any-unimported]
   myML,                                  # should be fitted model 
   Xdf: Optional[pd.DataFrame] = None,    # dataframe for permutation importance 
   ydf: Optional[pd.DataFrame] = None,
   n_repeats: Optional[pd.DataFrame] = None,   # number of permutation, default = 5
   random_state:  Optional[pd.DataFrame] = None,
   max_samples: Optional[float|int] = None    # default = 1
) -> Tuple[pd.DataFrame]:
   temp = None
   fields = myML.feature_names_in_
   if myML.__str__().find('RandomForestClassifier') >= 0 or myML.__str__().find('RandomForestRegressor') >= 0:
      trf = myML.feature_importances_
      temp = pd.DataFrame(zip(fields,trf))   # for pd.DataFrame  dict is possible as well
   else:   
      if Xdf is None or ydf is None:
          raise InvalidParameterValueException ('***  Estimator is not RandomForest, Xdf and ydf must be given')                                                                        # feature importance from Randmom forres
   if Xdf is not None and ydf is not None:      # Permutation feature importance
      if len(Xdf.columns) == 0:
         raise InvalidParameterValueException ('***  DataFrame has no column')
      if len(Xdf.index) == 0:
         raise InvalidParameterValueException ('***  DataFrame has no rows')
      t = permutation_importance(myML, Xdf, ydf, n_repeats=10, random_state=42, n_jobs=2)
      if temp is None:
         importance = pd.DataFrame(zip(fields,t.importances_mean,t.importances_std),columns=['feature','permutation mean','permutation std'])
      else:
         importance = pd.DataFrame(zip(fields,trf,t.importances_mean,t.importances_std),columns=['feature','RandomForest','permutation mean','permutation std'])
   return importance

# *******************************
def model_importance(  # type: ignore[no-any-unimported]
   myML, 
   Xdf: Optional[pd.DataFrame] = None,
   ydf: Optional[pd.DataFrame] = None,
   n_repeats: Optional[pd.DataFrame] = None,
   random_state:  Optional[pd.DataFrame] = None,
   max_samples: Optional[float|int] = None
) -> Tuple[pd.DataFrame,pd.DataFrame]: 

   """ 
      Calcultaes feature importance
      - without Xdf and ydf:   Importance for RanomForrestClassifier and Regressor
      - with Xdf and ydf:      Permutation importance - verry time consuming
   Args:
      - myML (model). even for comparison with a testset the model ist used to get the model-typ (regression or classification)
      - Xdf Pandas dataframe or numpy array ("array-like"): subset of X of training
      - ydf Pandas dataframe or numpy array ("array-like"): subset of y of training
      for Permutation importances:
      - n_repeats (int, default=5): Number of times to permute a feature:  higher number mean more time!
      - random_state (int, default=None): RandomState instance
         Pseudo-random number generator to control the permutations of each feature. 
         Pass an int to get reproducible results across function calls.
      - max_samples (int or float, default=1.0): The number of samples to draw from X to compute feature importance in each repeat (without replacement).
         - If int, then draw max_samples samples.
         - If float, then draw max_samples * X.shape[0] samples.  
         - If max_samples is equal to 1.0 or X.shape[0], all samples will be used.
         While using this option may provide less accurate importance estimates, it keeps the method tractable when evaluating feature importance on large datasets. In combination with n_repeats, this allows to control the computational speed vs statistical accuracy trade-off of this method.
   Returns:
      DataFrame:
         Importance of Random Forrest:   one columns
         Permutation importance:   colums: mean ans std
   """

   importance = _model_importance(
      myML = myML,
      Xdf = Xdf,
      ydf = ydf,
      n_repeats = n_repeats,
      random_state = random_state,
      max_samples = max_samples
   )

   return importance
