import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Literal, Optional, Sequence, Tuple, Union
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from eis_toolkit import exceptions
from eis_toolkit.prediction.model_utils import _train_and_evaluate_sklearn_model


@beartype
def random_forest_classifier_train(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    test_method: Literal["simple_split", "kfold_cv", "skfold_cv", "loo_cv", "none"] = "simple_split",
    metrics: Sequence[Literal["accuracy", "precision", "recall", "f1"]] = ["accuracy"],
    simple_split_size: float = 0.2,
    cv_folds: int = 5,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: Optional[int] = 42,
    **kwargs,
) -> Tuple[RandomForestClassifier, dict]:
    """
    Train and optionally validate a Random Forest classifier model using Sklearn.

    Various options and configurations for model performance evaluation are available. No validation,
    simple train-test and cross-validation can be chosen. If validation is performed, metric(s) to
    calculate can be defined and validation process configured (cross-validation method, number of folds,
    size of the simple train-test split). Depending on the details of the validation process, the output
    metrics dictionary can be empty, one-dimensional or nested.

    Args:
        X: Training data.
        y: Target labels.
        test_method: Test / validation method to use. "simple_split" divides data into two parts, "kfold_cv"
            performs k-fold cross-validation, "skfold_cv" performs stratified k-fold cross-validation,
            "loo_cv" performs leave-one-out cross-validation and "none" will not test / validate model at all
            (in this case, all X and y will be used solely for training).
        metrics: Metrics to use for scoring the model. Defaults to "accuracy".
        simple_split_size: Fraction of the dataset to be used as test data (rest is used for training).
            Used only when test_method is "simple_split". Defaults to 0.2.
        cv_folds: Number of folds used in cross-validation. Used only when test_method is "kfold_cv"
            or "skfold_cv". Defaults to 5.
        n_estimators: The number of trees in the forest. Defaults to 100.
        max_depth: The maximum depth of the tree. If None, then nodes are expanded until all leaves are
            pure or until all leaves contain less than min_samples_split samples. Defaults to None.
        random_state: Seed for random number generation. Defaults to 42.
        **kwargs: Additional parameters for Sklearn's RandomForestClassifier.

    Returns:
        The trained RandomForestClassifier and metric scores as a dictionary.

    Raises:
        InvalidParameterValueException: If some of the numeric parameters are given invalid input values.
    """
    if not n_estimators >= 1:
        raise exceptions.InvalidParameterValueException("N-estimators must be at least 1.")

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, **kwargs)

    model, metrics = _train_and_evaluate_sklearn_model(
        X=X,
        y=y,
        model=model,
        test_method=test_method,
        metrics=metrics,
        simple_split_size=simple_split_size,
        cv_folds=cv_folds,
        random_state=random_state,
    )

    return model, metrics


@beartype
def random_forest_regressor_train(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    test_method: Literal["simple_split", "kfold_cv", "skfold_cv", "loo_cv", "none"] = "simple_split",
    metrics: Sequence[Literal["mse", "rmse", "mae", "r2"]] = ["mse"],
    simple_split_size: float = 0.2,
    cv_folds: int = 5,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: Optional[int] = 42,
    **kwargs,
) -> Tuple[RandomForestRegressor, dict]:
    """
    Train and optionally validate a Random Forest regressor model using Sklearn.

    Various options and configurations for model performance evaluation are available. No validation,
    simple train-test and cross-validation can be chosen. If validation is performed, metric(s) to
    calculate can be defined and validation process configured (cross-validation method, number of folds,
    size of the simple train-test split). Depending on the details of the validation process, the output
    metrics dictionary can be empty, one-dimensional or nested.

    Args:
        X: Training data.
        y: Target labels.
        test_method: Test / validation method to use. "simple_split" divides data into two parts, "kfold_cv"
            performs k-fold cross-validation, "skfold_cv" performs stratified k-fold cross-validation,
            "loo_cv" performs leave-one-out cross-validation and "none" will not test / validate model at all
            (in this case, all X and y will be used solely for training).
        metrics: Metrics to use for scoring the model. Defaults to "mse".
        simple_split_size: Fraction of the dataset to be used as test data (rest is used for training).
            Used only when test_method is "simple_split". Defaults to 0.2.
        cv_folds: Number of folds used in cross-validation. Used only when test_method is "kfold_cv"
            or "skfold_cv". Defaults to 5.
        n_estimators: The number of trees in the forest. Defaults to 100.
        max_depth: The maximum depth of the tree. If None, then nodes are expanded until all leaves are
            pure or until all leaves contain less than min_samples_split samples. Defaults to None.
        random_state: Seed for random number generation. Defaults to 42.
        **kwargs: Additional parameters for Sklearn's RandomForestRegressor.

    Returns:
        The trained RandomForestRegressor and metric scores as a dictionary.

    Raises:
        InvalidParameterValueException: If some of the numeric parameters are given invalid input values.
    """
    if not n_estimators >= 1:
        raise exceptions.InvalidParameterValueException("N-estimators must be at least 1.")

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, **kwargs)

    model, metrics = _train_and_evaluate_sklearn_model(
        X=X,
        y=y,
        model=model,
        test_method=test_method,
        metrics=metrics,
        simple_split_size=simple_split_size,
        cv_folds=cv_folds,
        random_state=random_state,
    )

    return model, metrics
