
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from eis_toolkit.exceptions import InvalidParameterValueException, InvalideContentOfInputDataFrame

# *******************************

def _sklearn_model_crossvalidation(  # type: ignore[no-any-unimported]
    sklearnMl,
    Xdf: pd.DataFrame,                  # dataframe of Features for traning
    ydf: pd.DataFrame,                  # dataframe of known values for training
    scoring: Optional[str | list | tuple | dict] = None,
    cv: Optional[int] = None,           # int: number of the folds (default 5)
    n_jobs: Optional[int] = None,       # if None: complement size of the test_size
    verbose: Optional [int] = 0,
    pre_dispatch: Optional [int] = None,
    return_train_score: Optional [bool] = False,       # a list for Classification estimators: ['accuracy','recall_macro','precision_macro','f1_macro']
                                                    # a list of Regression estimators:  ['r2','explained_variance','neg_mean_absolute_error','neg_mean_squared_error']
) -> dict:

    ty = ydf
    if len(ydf.shape) > 1:
        if ydf.shape[1] == 1:
            ty = np.ravel(ydf)
    
    #  Crossvalidation
    # if not sklearnMl._estimator_type == 'regressor':
    #      if np.issubdtype(ty.dtype, np.floating):
    #         raise InvalideContentOfInputDataFrame('A classifier model cannot us a float y (target)')

    if sklearnMl._estimator_type == 'classifier':
        if np.issubdtype(ty.dtype, np.floating):
            raise InvalideContentOfInputDataFrame('A classifier model cannot us a float y (target)')
            #ty = (ty + 0.5).astype(np.uint16)
    else:
        if not np.issubdtype(ty.dtype, np.number):
            raise InvalideContentOfInputDataFrame('A regressor model can only use number y (target)')

    sklearnMl.fit(Xdf, ty)
    if scoring is None:
        if sklearnMl._estimator_type == 'regressor':
            scoring = ['r2','explained_variance','neg_mean_absolute_error','neg_mean_squared_error']
        else:
            scoring = ['accuracy','recall_macro','precision_macro','f1_macro']
    crsv = cross_validate(estimator = sklearnMl,
                        X = Xdf,
                        y = ty,
                        scoring = scoring,
                        cv = cv,
                        n_jobs = n_jobs,
                        verbose = verbose,
                        pre_dispatch = pre_dispatch,
                        return_train_score = return_train_score,
                        )

    return crsv

# *******************************
def sklearn_model_crossvalidation(  # type: ignore[no-any-unimported]
    sklearnMl,
    Xdf: pd.DataFrame,                      # dataframe of Features for traning
    ydf: pd.DataFrame,                      # dataframe of known values for training
    scoring: Optional[str | list | tuple | dict] = None,
    cv: Optional[int] = None,               # int: number of the folds (None -> 5)
    n_jobs: Optional[int] = None,           # if None: complement size of the test_size
    verbose: Optional [int] = 0,
    pre_dispatch: Optional [int] = None,
    return_train_score: Optional [bool] = None,       # a list for Classification estimators: ['accuracy','recall_macro','precision_macro','f1_macro']
                                                    # a list of Regression estimators:  ['r2','explained_variance','neg_mean_absolute_error','neg_mean_squared_error'] 
) -> dict:

    """ 
        Cross validation for a Sklearn model.
    Args:
        - Xdf Pandas dataframe or numpy array ("array-like"): features (columns) and samples (rows)
        - ydf Pandas dataframe or numpy array ("array-like"): target valus(columns) and samples (rows) (same number as Xdf)
            If ydf is float and the estimator is a classifier: ydf will be rounded to int.
        - scoring (str, list, tuple, dict; optional), 
            If scoring represents a single score, one can use: str
            If scoring represents multiple scores, one can use:
                - a list or tuple of unique strings:
                    for classification: ['accuracy','recall_macro','precision_macro','f1_macro']
                    for regression: ['r2','explained_variance','neg_mean_absolute_error','neg_mean_squared_error']
                - a dictionary with metric names as keys and callables a values.
        - cv (int, default = 4). cross-validation generator or an iterable,    
            Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 5-fold cross validation,
                int, to specify the number of folds in a (Stratified)KFold,
            CV splitter,
            An iterable yielding (train, test) splits as arrays of indices.
            For int/None inputs, if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used. 
            In all other cases, Fold is used. These splitters are instantiated with shuffle=False so the splits will be the same across calls.
        - n_jobs (int, default=None): Number of jobs to run in parallel. 
            Training the estimator and computing the score are parallelized over the cross-validation splits. 
            None means 1 unless in a joblib.parallel_backend context. 
            -1 means using all processors
        - verbose (int, default=0): The verbosity level.
        - pre_dispatch (int or str, default= 2*n_jobs): 
            Controls the number of jobs that get dispatched during parallel execution. Reducing this number can be useful to avoid an explosion of memory consumption when more jobs get dispatched than CPUs can process. 
            This parameter can be:
            None, in which case all the jobs are immediately created and spawned. Use this for lightweight and fast-running jobs, to avoid delays due to on-demand spawning of the jobs
            An int, giving the exact number of total jobs that are spawned
            A str, giving an expression as a function of n_jobs, as in 2*n_jobs
   Returns:
        Dictionary of the reached scores
            scores dict of float arrays of shape (n_splits,)
            Array of scores of the estimator for each run of the cross validation.
            A dict of arrays containing the score/time arrays for each scorer is returned. The possible keys for this dict are:
            test_score
                The score array for test scores on each cv split. Suffix _score in test_score changes to a specific metric like test_r2 or test_auc if there are multiple scoring metrics in the scoring parameter.
            train_score
                The score array for train scores on each cv split. Suffix _score in train_score changes to a specific metric like train_r2 or train_auc if there are multiple scoring metrics in the scoring parameter. This is available only if return_train_score parameter is True.
            fit_time
                The time for fitting the estimator on the train set for each cv split.
            score_time
                The time for scoring the estimator on the test set for each cv split. (Note time for scoring on the train set is not included even if return_train_score is set to True
    """

    # Argument evaluation
    fl = []
    t = sklearnMl.__class__.__name__           #t = isinstance(sklearnMl,(RandomForestClassifier,RandomForestRegressor,LogisticRegression))
    if not t in ('RandomForestClassifier','RandomForestRegressor','LogisticRegression'):
        fl.append('Argument sklearnMl is not an instance of one of (RandomForestClassifier,RandomForestRegressor,LogisticRegression)')
    if not (isinstance(Xdf, pd.DataFrame)):
        fl.append('Argument Xdf is not a DataFrame')
    if not (isinstance(ydf, pd.DataFrame)):
        fl.append('Argument ydf is not a DataFrame')
    if not (isinstance(cv, int) or (cv is None)):
        fl.append('Argument cv is not integer and is not None')
    if not (isinstance(n_jobs, int) or (n_jobs is None)):
        fl.append('Argument n_jobs is not integer and is not None')
    if not (isinstance(verbose, int) or (verbose is None)):
        fl.append('Argument verbose is not integer and is not None')
    if not (isinstance(pre_dispatch, int) or (pre_dispatch is None)):
        fl.append('Argument pre_dispatch is not integer and is not None')
    if not (isinstance(return_train_score ,bool) or (return_train_score is None)):
        fl.append('Argument return_train_score is not bool and is not None')
    if len(fl) > 0:
        raise InvalidParameterValueException(fl[0])
    
    fl = []
    if len(Xdf.columns) == 0:
        fl.append('DataFrame Xdf has no column')
    if len(Xdf.index) == 0:
        fl.append('DataFrame Xdf has no rows')
    if len(ydf.columns) != 1:
        fl.append('DataFrame ydf has 0 or more then 1 columns')
    if len(ydf.index) == 0:
        fl.append('DataFrame ydf has no rows')
    if Xdf.isna().sum().sum() > 0 or ydf.isna().sum().sum() > 0:
        fl.append('DataFrame train_ydf or train_Xdf contains Nodata-values')    
    if len(fl) > 0:
        raise InvalideContentOfInputDataFrame(fl[0])

    # check cv:
    if cv == None: 
        tmp = 5
    else:
        tmp = cv
    if Xdf.shape[0] / tmp < 4:
         raise InvalideContentOfInputDataFrame('cross validation splitting: X to smal / cv to hight ') 

    return _sklearn_model_crossvalidation(
        sklearnMl = sklearnMl,
        Xdf = Xdf, 
        ydf = ydf,
        scoring = scoring,
        cv = cv,
        n_jobs = n_jobs,
        verbose = verbose,
        pre_dispatch = pre_dispatch,
        return_train_score = return_train_score,
    )


