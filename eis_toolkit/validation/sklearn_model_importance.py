
from typing import Optional, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************

def _sklearn_model_importance(  # type: ignore[no-any-unimported]
   sklearnMl,                                  # should be fitted model 
   Xdf: Optional[pd.DataFrame] = None,    # dataframe for permutation importance 
   ydf: Optional[pd.DataFrame] = None,
   #scoring: Optional[list] = None,
   n_repeats: Optional[int] = None,   # number of permutation, default = 5
   random_state:  Optional[int] = None,
   n_jobs: Optional[int] = None, 
   max_samples: Optional[float|int] = None,   # default = 1
) -> Tuple[pd.DataFrame]:
   
      # Argument evaluation
   fl = []
   t = sklearnMl.__class__.__name__           #t = isinstance(sklearnMl,(RandomForestClassifier,RandomForestRegressor,LogisticRegression))
   if not t in ('RandomForestClassifier','RandomForestRegressor','LogisticRegression'):
      fl.append('argument sklearnMl is not an instance of one of (RandomForestClassifier,RandomForestRegressor,LogisticRegression)')
   if not (isinstance(Xdf,pd.DataFrame) or (Xdf is None)):
      fl.append('argument Xdf is not a DataFrame and is not None')
   if not (isinstance(ydf,pd.DataFrame)  or (ydf is None)):
      fl.append('argument ydf is not a DataFrame and is not None')
   # if not (isinstance(scoring,list) or (scoring is None)):
   #    fl.append('argument scoring is not integer and is not None')
   if not (isinstance(n_jobs,int) or (n_jobs is None)):
      fl.append('argument n_jobs is not integer and is not None')
   if not (isinstance(random_state,int) or (random_state is None)):
      fl.append('argument random_state is not integer and is not None')
   if not (isinstance(n_repeats,int) or (n_repeats is None)):
      fl.append('argument n_repeats is not bool and is not None')
   if not (isinstance(max_samples,(float,int)) or (max_samples is None)):
      fl.append('argument max_samples is not integer or float and is not None')   
   if len(fl) > 0:
      raise InvalidParameterValueException ('***  function sklearn_model_importance: ' + fl[0])

   temp = None 
   fields = sklearnMl.feature_names_in_
   if sklearnMl.__str__().find('RandomForestClassifier') >= 0 or sklearnMl.__str__().find('RandomForestRegressor') >= 0:
      trf = sklearnMl.feature_importances_
      temp = pd.DataFrame(zip(fields,trf))   # for pd.DataFrame  dict is possible as well
   else:
      if Xdf is None or ydf is None:
          raise InvalidParameterValueException ('***  function sklearn_model_importance: Estimator is not RandomForest, Xdf and ydf must be given')                                                                        # feature importance from Randmom forres
   if Xdf is not None and ydf is not None:      # Permutation feature importance
      if len(Xdf.columns) == 0:
         raise InvalidParameterValueException ('***  function sklearn_model_importance: DataFrame Xdf has no column')
      if len(Xdf.index) == 0:
         raise InvalidParameterValueException ('***  function sklearn_model_importance: DataFrame Xdf has no rows')

      # if scoring is None:
      #    if sklearnMl._estimator_type == 'regressor':
      #          scoring = ['r2','explained_variance','neg_mean_absolute_error','neg_mean_squared_error']
      #    else: 
      #          scoring = ['accuracy','recall_macro','precision_macro','f1_macro']
      
      t = permutation_importance(sklearnMl,Xdf,ydf,n_repeats=n_repeats,random_state=random_state,n_jobs=n_jobs)  #,scoring=scoring
      if temp is None:
         importance = pd.DataFrame(zip(fields,t.importances_mean,t.importances_std),columns=['feature','permutation mean','permutation std'])
      else:
         importance = pd.DataFrame(zip(fields,trf,t.importances_mean,t.importances_std),columns=['feature','RandomForest','permutation mean','permutation std'])
   return importance

# *******************************
def sklearn_model_importance(  # type: ignore[no-any-unimported]
   sklearnMl, 
   Xdf: Optional[pd.DataFrame] = None,
   ydf: Optional[pd.DataFrame] = None,
   #scoring: Optional[list] = None,
   n_repeats: Optional[int] = None,   # number of permutation, default = 5
   random_state:  Optional[int] = None,
   n_jobs: Optional[int] = None, 
   max_samples: Optional[float|int] = None,   # default = 1
) -> Tuple[pd.DataFrame,pd.DataFrame]: 

   """ 
      Calcultaes feature importance
      - without Xdf and ydf:   Importance for RanomForrestClassifier and Regressor
      - with Xdf and ydf:      Permutation importance - verry time consuming
   Args:
      - sklearnMl (model). even for comparison with a testset the model ist used to get the model-typ (regression or classification)
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

   importance = _sklearn_model_importance(
      sklearnMl = sklearnMl,
      Xdf = Xdf,
      ydf = ydf,
      #scoring = scoring,
      n_repeats = n_repeats,
      random_state = random_state,
      n_jobs = n_jobs,
      max_samples = max_samples,
   )

   return importance
