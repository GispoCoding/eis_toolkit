
from typing import Optional, Literal
from sklearn.ensemble import RandomForestRegressor
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************
MODE = Literal['squared_error','absolute_error','poisson']
maxf = Literal['sqrt','log2',None]
def _randomforest_regressor(  # type: ignore[no-any-unimported]
   n_estimators: Optional[int] = 100 ,
   criterion: Optional [MODE] = 'squered_error',
   max_depth: Optional[int | float] = None,
   min_samples_split: Optional[int | float] = 2,
   min_samples_leaf: Optional[int] = 1,        
   min_weight_fraction_leaf: Optional[float] = 0.0,
   max_features: Optional[maxf | int | float]  = 1.0,
   max_leaf_nodes: Optional[int] = None,
   min_impurity_decrease: Optional[float]  = 0.0,
   bootstrap: Optional[bool] = True,
   oob_score: Optional[bool] = False,
   n_jobs: Optional[int] = None,
   random_state: Optional [int] = None,
   verbose: Optional [int] = 0,
   warm_start: Optional [bool] = False,
   ccp_alpha: Optional [float] = 0.0,
   max_samples: Optional [int | float] = None
):

   myML = RandomForestRegressor(
      n_estimators = n_estimators,
      criterion = criterion,
      max_depth = max_depth,
      min_samples_split = min_samples_split,
      min_samples_leaf = min_samples_leaf,
      min_weight_fraction_leaf = min_weight_fraction_leaf,
      max_features = max_features,
      max_leaf_nodes = max_leaf_nodes,
      min_impurity_decrease  = min_impurity_decrease,
      bootstrap = bootstrap,
      oob_score = oob_score,
      n_jobs = n_jobs,
      random_state = random_state,
      verbose = verbose,
      warm_start = warm_start,
      ccp_alpha = ccp_alpha,
      max_samples = max_samples
)
   return myML

# *******************************
MODE = Literal['squared_error','absolute_error','poisson']
maxf = Literal['sqrt','log2',None]
def randomforest_regressor(  # type: ignore[no-any-unimported]
   n_estimators: Optional[int] = 100,
   criterion: Optional [MODE] = 'squared_error',
   max_depth: Optional[int | float] = None,
   min_samples_split: Optional[int | float] = 2,
   min_samples_leaf: Optional[int] = 1,        
   min_weight_fraction_leaf: Optional[float] = 0.0,
   max_features: Optional[maxf | int | float]  = 1.0,
   max_leaf_nodes: Optional[int] = None,
   min_impurity_decrease: Optional[float]  = 0.0,
   bootstrap: Optional[bool] = True,
   oob_score: Optional[bool] = False,
   n_jobs: Optional[int] = None,
   random_state: Optional [int] = None,
   verbose: Optional [int] = 0,
   warm_start: Optional [bool] = False,
   ccp_alpha: Optional [float] = 0.0,
   max_samples: Optional [int | float] = None
):

    """ 
      training of a randmon forest model
    Args:
      - n_estimators (int, default=100):    The number of trees in the forest.
      - criterion ({“squared_error”, “absolute_error”, “poisson”}, default=”squared_error”): 
         the function to measure the quality of a split. 
         Supported criteria are “squared_error” for the mean squared error, which is equal to variance reduction as feature selection criterion, 
         “absolute_error” for the mean absolute error, and “poisson” which uses reduction in Poisson deviance 
         to find splits. Training using “absolute_error” is significantly slower than when using “squared_error”.
      - max_depth (int, default=None):  The maximum depth of the tree. 
         If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
      - min_samples_split (int or float, default=2):  The minimum number of samples required to split an internal node:
         If int, then consider min_samples_split as the minimum number.
         If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.
      - min_samples_leaf (int or float, default=1): The minimum number of samples required to be at a leaf node. 
         A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. 
         This may have the effect of smoothing the model, especially in regression.
         If int, then consider min_samples_leaf as the minimum number.
         If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
      - min_weight_fraction_leaf (float, default=0.0): The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. 
         Samples have equal weight when sample_weight is not provided.
      - max_features ({“sqrt”, “log2”, None}, int or float, default=1.0): The number of features to consider when looking for the best split:
         If int, then consider max_features features at each split.
         If float, then max_features is a fraction and max(1, int(max_features * n_features_in_)) features are considered at each split.
         If “auto”, then max_features=n_features.
         If “sqrt”, then max_features=sqrt(n_features).
         If “log2”, then max_features=log2(n_features).
         If None or 1.0, then max_features=n_features.
         Note The default of 1.0 is equivalent to bagged trees and more randomness can be achieved by setting smaller values, e.g. 0.3.
         Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.
      - max_leaf_nodes (int, default=None): Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
      - min_impurity_decrease (float, default=0.0): A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
         The weighted impurity decrease equation is the following:
         N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)
         where N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the number of samples in the left child, and N_t_R is the number of samples in the right child.
         N, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight is passed.
      - bootstrap (bool, default=True):  Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
      - oob_score (bool, default=False): Whether to use out-of-bag samples to estimate the generalization score. Only available if bootstrap=True.
    
      not implmented: 
      - n_jobs int, default=None: The number of jobs to run in parallel. fit, predict, decision_path and apply are all parallelized over the trees. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.
      - random_state int, RandomState instance or None, default=None: Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features). See Glossary for details.
      - verbose int, default=0: Controls the verbosity when fitting and predicting.
      - warm_start bool, default=False: When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest. See the Glossary.
      - ccp_alphanon  negative float, default=0.0: Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed. See Minimal Cost-Complexity Pruning for details.
      - max_samples  int or float, default=None: If bootstrap is True, the number of samples to draw from X to train each base estimator.
         If None (default), then draw X.shape[0] samples.
         If int, then draw max_samples samples.
         If float, then draw max_samples * X.shape[0] samples. Thus, max_samples should be in the interval (0.0, 1.0].
    Returns:
        randomforrest model
    """

    myML = _randomforest_regressor( 
      n_estimators = n_estimators,
      criterion = criterion,
      max_depth = max_depth,
      min_samples_split = min_samples_split,
      min_samples_leaf = min_samples_leaf,
      min_weight_fraction_leaf = min_weight_fraction_leaf,
      max_features = max_features,
      max_leaf_nodes = max_leaf_nodes,
      min_impurity_decrease  = min_impurity_decrease,
      bootstrap = bootstrap,
      oob_score = oob_score,
      n_jobs = n_jobs,
      random_state = random_state,
      verbose = verbose,
      warm_start = warm_start,
      ccp_alpha = ccp_alpha,
      max_samples = max_samples
    )

    return myML


