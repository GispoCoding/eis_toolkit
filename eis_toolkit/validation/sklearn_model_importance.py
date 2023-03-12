
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from eis_toolkit.exceptions import InvalidParameterValueException, ModelIsNotFitted, InvalideContentOfInputDataFrame

# *******************************

def _sklearn_model_importance(  # type: ignore[no-any-unimported]
   sklearnMl,                                  # should be fitted model 
   Xdf: Optional[pd.DataFrame] = None,    # dataframe for permutation importance 
   ydf: Optional[pd.DataFrame] = None,
   scoring: Optional[str | list | tuple | dict] = None,
   n_repeats: Optional[int] = 5,   # number of permutation, default = 5
   random_state:  Optional[int] = None,
   n_jobs: Optional[int] = None,
   max_samples: Optional[float|int] = 1.0,   # default = 1.0
) -> pd.DataFrame:

   importance = None 
   fields = sklearnMl.feature_names_in_
   if sklearnMl.__str__().find('RandomForestClassifier') >= 0 or sklearnMl.__str__().find('RandomForestRegressor') >= 0:
      trf = sklearnMl.feature_importances_
      importance = pd.DataFrame(zip(fields, trf))   # for pd.DataFrame  dict is possible as well
   else:
      if Xdf is None or ydf is None:
          raise InvalidParameterValueException('estimator is not RandomForest, Xdf and ydf must be given')                                                                        # feature importance from Randmom forres
   if Xdf is not None and ydf is not None:      # Permutation feature importance
      fl = []
      if len(Xdf.columns) == 0:
         fl.append('DataFrame Xdf has no column')
      if len(Xdf.index) == 0:
         fl.append('DataFrame Xdf has no rows')
      if len(ydf.columns) != 1:
         fl.append('DataFrame ydf has 0 or more then 1 columns')
      if len(ydf.index) == 0:
         fl.append('DataFrame ydf has no rows')
      if len(fl) > 0:
         raise InvalideContentOfInputDataFrame(fl[0])

      # if scoring is None:
      #    if sklearnMl._estimator_type == 'regressor':
      #          scoring = ['r2','explained_variance','neg_mean_absolute_error','neg_mean_squared_error']
      #    else: 
      #          scoring = ['accuracy','recall_macro','precision_macro','f1_macro']
         
      if sklearnMl._estimator_type == 'classifier':
         if np.issubdtype(ydf.dtypes[0], np.floating):
            raise InvalideContentOfInputDataFrame('A classifier model cannot us a float y (target)')
            #ydf = (ydf + 0.5).astype(np.uint16)
      else:
         if not np.issubdtype(ydf.dtypes[0], np.number):
            raise InvalideContentOfInputDataFrame('A regressor model can only use number y (target)')
      
      if Xdf.isna().sum().sum() > 0 or ydf.isna().sum().sum() > 0:
         raise InvalideContentOfInputDataFrame('DataFrame ydf or Xdf contains Nodata-values')
      
      t = permutation_importance(
         sklearnMl, 
         Xdf, 
         ydf, 
         n_repeats=n_repeats, 
         random_state=random_state, 
         n_jobs=n_jobs, 
         max_samples=max_samples, 
         scoring=scoring
         )
      if importance is None:
         importance = pd.DataFrame(zip(fields, 
                                       t.importances_mean, 
                                       t.importances_std), 
                                       columns=['feature','permutation mean','permutation std'])
      else:
         importance = pd.DataFrame(zip(fields, 
                                       trf, 
                                       t.importances_mean, 
                                       t.importances_std), 
                                       columns=['feature','RandomForest','permutation mean','permutation std'])
   return importance

# *******************************
def sklearn_model_importance(  # type: ignore[no-any-unimported]
   sklearnMl, 
   Xdf: Optional[pd.DataFrame] = None,
   ydf: Optional[pd.DataFrame] = None,
   scoring: Optional[str | list | tuple | dict] = None,
   n_repeats: Optional[int] = 5,   # number of permutation, default = 5
   random_state:  Optional[int] = None,
   n_jobs: Optional[int] = None, 
   max_samples: Optional[float|int] = 1.0,   # default = 1.0
) -> pd.DataFrame: 

   """ 
      Calcultaes feature importance
      - without Xdf and ydf:   Importance for RanomForrestClassifier and Regressor
      - with Xdf and ydf:      Permutation importance - verry time consuming
   Args:
      - sklearnMl (model): Even for comparison with a testset the model is used to get the model-typ (regression or classification)
      - Xdf Pandas dataframe or numpy array ("array-like"): subset of X of training
      - ydf Pandas dataframe or numpy array ("array-like"): subset of y of training
      for Permutation importances:
      - scoring (str, list, tuple, dict; optional), 
            If scoring represents a single score, one can use: str
            If scoring represents multiple scores, one can use:
                - a list or tuple of unique strings:
                    for classification: ['accuracy','recall_macro','precision_macro','f1_macro']
                    for regression: ['r2','explained_variance','neg_mean_absolute_error','neg_mean_squared_error']
                - a dictionary with metric names as keys and callables a values.
      - n_repeats (int, default=5): Number of times to permute a feature:  higher number mean more time!
      - random_state (int, default=None): RandomState instance
         Pseudo-random number generator to control the permutations of each feature. 
         Pass an int to get reproducible results across function calls.
      - max_samples (int or float, default=1.0): The number of samples to draw from X to compute feature importance in each repeat (without replacement).
         - If int, then draw max_samples samples.
         - If float, then draw max_samples * X.shape[0] samples.  
         - If max_samples is equal to 1.0 or X.shape[0], all samples will be used.
         While using this option may provide less accurate importance estimates, it keeps the method tractable when evaluating feature importance on large datasets. 
         In combination with n_repeats, this allows to control the computational speed vs statistical accuracy trade-off of this method.
   Returns:
      DataFrame:
         Importance of Random Forrest:   1 column
         Permutation importance:   2 colums: mean and std
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
   if not (isinstance(scoring, list) or isinstance(scoring,tuple) or isinstance(scoring,dict) or isinstance(scoring,str) or (scoring is None)):   
      fl.append('Argument scoring is not integer and is not None')
   if not (isinstance(n_jobs, int) or (n_jobs is None)):
      fl.append('Argument n_jobs is not integer and is not None')
   if not (isinstance(random_state, int) or (random_state is None)):
      fl.append('Argument random_state is not integer and is not None')
   if not (isinstance(n_repeats, int) or (n_repeats is None)):
      fl.append('Argument n_repeats is not bool and is not None')
   if not (isinstance(max_samples, (float,int)) or (max_samples is None)):
      fl.append('Argument max_samples is not integer or float and is not None')   
   if len(fl) > 0:
      raise InvalidParameterValueException (fl[0])
   
   if Xdf is not None:
      if ydf is None:
         raise InvalidParameterValueException('Xdf is not Non but ydf is None')
      else: 
         if Xdf.shape[0] != ydf.shape[0]:
            raise InvalideContentOfInputDataFrame('Xdf and ydf have not the same number of rows')

   if not hasattr(sklearnMl,'feature_names_in_'):
      raise ModelIsNotFitted('Model is not fitted')

   return  _sklearn_model_importance(
      sklearnMl = sklearnMl,
      Xdf = Xdf,
      ydf = ydf,
      scoring = scoring,
      n_repeats = n_repeats,
      random_state = random_state,
      n_jobs = n_jobs,
      max_samples = max_samples,
   )

