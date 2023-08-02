import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Any, Optional
from sklearn.model_selection import cross_validate

from eis_toolkit.exceptions import InvalidParameterValueException


# *******************************
@beartype
def _sklearn_model_crossvalidation(
    sklearnMl: Any,
    Xdf: pd.DataFrame,  # dataframe of Features for traning
    ydf: pd.DataFrame,  # dataframe of known values for training
    scoring: Optional[list] = None,
    cv: Optional[int] = None,  # int: number of the folds (default 5)
    n_jobs: Optional[int] = None,  # if None: complement size of the test_size
    verbose: Optional[int] = 0,
    pre_dispatch: Optional[int] = None,
    return_train_score: Optional[bool] = False,  
    # a list for Classification estimators: ['accuracy','recall_macro','precision_macro','f1_macro']
    # a list of Regression estimators:  ['r2','explained_variance','neg_mean_absolute_error','neg_mean_squared_error']
) -> dict:

    ty = ydf
    if len(ydf.shape) > 1:
        if ydf.shape[1] == 1:
            ty = np.ravel(ydf)

    #  Crossvalidation
    if sklearnMl._estimator_type == "classifier":
        if np.issubdtype(ty.dtype, np.floating):
            raise InvalidParameterValueException("A classifier model cannot us a float y (target)")
    else:
        if not np.issubdtype(ty.dtype, np.number):
            raise InvalidParameterValueException("A regressor model can only use number y (target)")

    sklearnMl.fit(Xdf, ty)

    if scoring is None:
        if sklearnMl._estimator_type == "regressor":
            scoring = [
                "r2",
                "explained_variance",
                "neg_mean_absolute_error",
                "neg_mean_squared_error",
            ]
        else:
            scoring = [
                "accuracy",
                "recall_macro",
                "precision_macro",
                "f1_macro",
            ]
    else:
        if sklearnMl._estimator_type == "regressor":
            a = [
                "explained_variance",
                "max_error",
                "neg_mean_absolute_error",
                "neg_mean_squared_error",
                "neg_root_mean_squared_error",
                "neg_mean_squared_log_error",
                "neg_median_absolute_error",
                "r2",
                "neg_mean_poisson_deviance",
                "neg_mean_gamma_deviance",
                "neg_mean_absolute_percentage_error",
                "d2_absolute_error_score",
                "d2_pinball_score",
                "d2_tweedie_score",
            ]
            if not (set(scoring) <= set(a)):
                raise InvalidParameterValueException("The list of the scores does not fit to regression")
        else:
            a = [
                "accuracy",
                "Cbalanced_accuracy",
                "top_k_accuracy",
                "average_precision",
                "neg_brier_score",
                "f1",
                "f1_micro",
                "f1_macro",
                "f1_weighted",
                "f1_samples",
                "neg_log_loss",
                "precision",
                "precision_micro",
                "recall",
                "recall_micro",
                "jaccard",
                "jaccard_micro",
                "precision_macro",
                "recall_macro",
                "jaccard_macro",
                "precision_weighted",
                "recall__weighted",
                "jaccard__weighted",
                "precision_samples",
                "recall___samples",
                "jaccard___samples",
                "roc_auc",
                "roc_auc_ovr",
                "roc_auc_ovo",
                "roc_auc_ovr_weighted",
                "roc_auc_ovo_weighted",
            ]
            if not (set(scoring) <= set(a)):
                raise InvalidParameterValueException("The list of the scores does not fit to classification")

    crsv = cross_validate(
        estimator=sklearnMl,
        X=Xdf,
        y=ty,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        pre_dispatch=pre_dispatch,
        return_train_score=return_train_score,
    )

    return crsv


# *******************************
@beartype
def sklearn_model_crossvalidation(  # type: ignore[no-any-unimported]
    sklearnMl: Any,
    Xdf: pd.DataFrame,  # dataframe of Features for traning
    ydf: pd.DataFrame,  # dataframe of known values for training
    scoring: Optional[list] = None,
    cv: Optional[int] = None,  # int: number of the folds (None -> 5)
    n_jobs: Optional[int] = None,  # if None: complement size of the test_size
    verbose: Optional[int] = 0,
    pre_dispatch: Optional[int] = None,
    return_train_score: Optional[bool] = None,  
    # a list for Classification estimators: ['accuracy','recall_macro','precision_macro','f1_macro']
    # a list of Regression estimators:  ['r2','explained_variance','neg_mean_absolute_error','neg_mean_squared_error']
) -> dict:

    """
        Cross validation for a Sklearn model.
    Args:
        - sklearnMl (model). The Model will be fitted based on the training dataset.
          Even for comparison with a testset (or verification dataset) the model is used to get the model-typ (regression or classification).
        - Xdf ("array-like"): features (columns) and samples (rows)
        - ydf ("array-like"): target valus(columns) and samples (rows) (same number as Xdf)
            In case the estimator is a classifier ydf should be int.
        - scoring,
            scoring list must be a subset of:
                for classification: ['accuracy', 'balanced_accuracy', 'top_k_accuracy', 'average_precision',
                                    'neg_brier_score', 'f1', 'f1_micro', 'f1_macro', 'f1_weighted', 'f1_samples',
                                    'neg_log_loss', 'precision', 'recall', 'jaccard', 'roc_auc', 'roc_auc_ovr',
                                    'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']
                for regression: ['explained_variance', 'max_error', 'neg_mean_absolute_error',
                                'neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_mean_squared_log_error',
                                'neg_median_absolute_error', 'r2', 'neg_mean_poisson_deviance',
                                'neg_mean_gamma_deviance', 'neg_mean_absolute_percentage_error',
                                'd2_absolute_error_score', 'd2_pinball_score', 'd2_tweedie_score']
            If scoring is None the following list is used
                for classification: ['accuracy','recall_macro','precision_macro','f1_macro']
                for regression: ['r2','explained_variance','neg_mean_absolute_error','neg_mean_squared_error']
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
        - return_train_score: Computing training scores is used to get insights on how different parameter settings impact the overfitting/underfitting trade-off. However computing the scores on the training set can be computationally expensive and is not strictly required to select the parameters that yield the best generalization performance.
    Returns:
        Dictionary of the reached scores
            scores dict of float arrays of shape (n_splits,)
            Array of scores of the estimator for each run of the cross validation.
            A dict of arrays containing the score/time arrays for each scorer is returned. The possible keys for this dict are:
            - test_score
                The score array for test scores on each cv split. Suffix _score in test_score changes to a specific metric like test_r2 or test_auc if there are multiple scoring metrics in the scoring parameter.
            - train_score
                The score array for train scores on each cv split. Suffix _score in train_score changes to a specific metric like train_r2 or train_auc if there are multiple scoring metrics in the scoring parameter. This is available only if return_train_score parameter is True.
            - fit_time
                The time for fitting the estimator on the train set for each cv split.
            - score_time
                The time for scoring the estimator on the test set for each cv split. (Note time for scoring on the train set is not included even if return_train_score is set to True
    """

    # Argument evaluation
    t = (
        sklearnMl.__class__.__name__
    )
    if not t in ("RandomForestClassifier", "RandomForestRegressor", "LogisticRegression"):
        raise InvalidParameterValueException(
            "Argument sklearnMl is not an instance of one of (RandomForestClassifier,RandomForestRegressor,LogisticRegression)"
        )
    if len(Xdf.columns) == 0:
        raise InvalidParameterValueException("DataFrame Xdf has no column")
    if len(Xdf.index) == 0:
        raise InvalidParameterValueException("DataFrame Xdf has no rows")
    if len(ydf.columns) != 1:
        raise InvalidParameterValueException("DataFrame ydf has 0 or more then 1 columns")
    if len(ydf.index) == 0:
        raise InvalidParameterValueException("DataFrame ydf has no rows")
    if Xdf.isna().sum().sum() > 0 or ydf.isna().sum().sum() > 0:
        raise InvalidParameterValueException("DataFrame train_ydf or train_Xdf contains Nodata-values")

    # check cv:
    if cv == None:
        tmp = 5
    else:
        tmp = cv
    if Xdf.shape[0] / tmp < 4:
        raise InvalidParameterValueException("cross validation splitting: X to smal / cv to hight ")

    return _sklearn_model_crossvalidation(
        sklearnMl=sklearnMl,
        Xdf=Xdf,
        ydf=ydf,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        pre_dispatch=pre_dispatch,
        return_train_score=return_train_score,
    )
