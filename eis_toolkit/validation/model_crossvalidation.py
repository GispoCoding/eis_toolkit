
from typing import Optional,Any
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from eis_toolkit.exceptions import InvalidParameterValueException

# *******************************

def _model_crossvalidation(  # type: ignore[no-any-unimported]
    myML: Any,
    Xdf: pd.DataFrame,                  # dataframe of Features for traning
    ydf: Optional[pd.DataFrame] = None, # dataframe of known values for training
    fields: Optional[dict] = None,
    scoring: Optional[list] = None,
    cv: Optional[int] = None,           # int: number of the folds (default 5)
    n_jobs: Optional[int] = None,       # if None: complement size of the test_size
    verbose: Optional [int] = 0,
    pre_dispatch: Optional [int | str] = '2*n_jobs',
    return_train_score: Optional [list] = True       # a list for Classification estimators: ['accuracy','recall_macro','precision_macro','f1_macro']
                                                    # a list of Regression estimators:  ['r2','explained_variance','neg_mean_absolute_error','neg_mean_squared_error']
) -> dict:

    # Check
    if len(Xdf.columns) == 0:
        raise InvalidParameterValueException ('***  DataFrame has no column')
    if len(Xdf.index) == 0:
        raise InvalidParameterValueException ('***  DataFrame has no rows')
    # check cv:
    if cv == None: 
        tmp = 5
    else:
        tmp = cv
    if Xdf.shape[0] / tmp < 4:
         raise InvalidParameterValueException ('***  cross validation splitting: X to smal / cv to hight ') 

    # if ydf not as an separated datafram: separat "t"-column out of Xdf
    if ydf is None:
        if fields is None:
            raise InvalidParameterValueException ('***  target and target-field are None: ') 
        else:
            name = {i for i in fields if fields[i]=="t"}
            ydf = Xdf[list(name)]
            Xdf.drop(name,axis=1,inplace=True)

    # ty = ydf
    # if len(ydf.shape) > 1:
    #     if ydf.shape[1] == 1:
    #         ty = np.ravel(ydf)

    #  Crossvalidation
    myML.fit(Xdf, ydf)
    if scoring is None:
        if myML._estimator_type == 'regressor':
            scoring = ['r2','explained_variance','neg_mean_absolute_error','neg_mean_squared_error']
        else: 
            scoring = ['accuracy','recall_macro','precision_macro','f1_macro']
    crsv = cross_validate(estimator = myML,
                        X = Xdf,
                        y = ydf,
                        scoring = scoring,
                        cv = cv,
                        n_jobs = n_jobs,
                        verbose = verbose,
                        pre_dispatch = pre_dispatch,
                        return_train_score = return_train_score,
                        )

    return crsv

# *******************************
def model_crossvalidation(  # type: ignore[no-any-unimported]
    myML: Any,
    Xdf: pd.DataFrame,                      # dataframe of Features for traning
    ydf: Optional[pd.DataFrame] = None,      # dataframe of known values for training
    fields: Optional[dict] = None,
    scoring: Optional[str | list | tuple | dict] = None,
    cv: Optional[int] = None,               # int: number of the folds (None -> 5)
    n_jobs: Optional[int] = None,           # if None: complement size of the test_size
    verbose: Optional [int] = 0,
    pre_dispatch: Optional [int | str] = '2*n_jobs',
    return_train_score: Optional [list] = True       # a list for Classification estimators: ['accuracy','recall_macro','precision_macro','f1_macro']
                                                    # a list of Regression estimators:  ['r2','explained_variance','neg_mean_absolute_error','neg_mean_squared_error'] 
) -> dict:

    """ 
        cross validation for a ML model
    Args:
        - Xdf Pandas dataframe or numpy array ("array-like"): features (columns) and samples (rows)
        - ydf Pandas dataframe or numpy array ("array-like"): target valus(columns) and samples (rows) (same number as Xdf)
            If ydf is = None, target column is included in Xdf. In this case fields should not be None
        - fields (listdictionary default = None): the fieldnames and type of fields. A field type 't' is needed, fields is not needed if ydf is not None.
        - scoringstr (str, list, tuple, dict default=None), 
            If scoring represents a single score, one can use: str
            If scoring represents multiple scores, one can use:
                - a list or tuple of unique strings:
                    for classification: ['accuracy','recall_macro','precision_macro','f1_macro']
                    for regression: ['r2','explained_variance','neg_mean_absolute_error','neg_mean_squared_error']
                - a dictionary with metric names as keys and callables a values.
        - cv (int, default=None). cross-validation generator or an iterable,    
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
        - pre_dispatch (int or str, default=’2*n_jobs’): 
            Controls the number of jobs that get dispatched during parallel execution. Reducing this number can be useful to avoid an explosion of memory consumption when more jobs get dispatched than CPUs can process. 
            This parameter can be:
            None, in which case all the jobs are immediately created and spawned. Use this for lightweight and fast-running jobs, to avoid delays due to on-demand spawning of the jobs
            An int, giving the exact number of total jobs that are spawned
            A str, giving an expression as a function of n_jobs, as in ‘2*n_jobs’
   Returns:
        Dictionary of the reached scores    
            scoresdict of float arrays of shape (n_splits,)
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

    return _model_crossvalidation(
        myML = myML,
        Xdf = Xdf, 
        ydf = ydf, 
        fields = fields,
        scoring = scoring,     
        cv = cv,
        n_jobs = n_jobs,
        verbose = verbose,
        pre_dispatch = pre_dispatch,
        return_train_score = return_train_score
    )


