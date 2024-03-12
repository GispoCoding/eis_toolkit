import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Literal, Optional, Sequence, Tuple, Union
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.prediction.machine_learning_general import _train_and_validate_sklearn_model


@beartype
def random_forest_classifier_train(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    validation_method: Literal["split", "kfold_cv", "skfold_cv", "loo_cv", "none"] = "split",
    metrics: Sequence[Literal["accuracy", "precision", "recall", "f1"]] = ["accuracy"],
    split_size: float = 0.2,
    cv_folds: int = 5,
    n_estimators: int = 100,
    criterion: Literal["gini", "entropy", "log_loss"] = "gini",
    max_depth: Optional[int] = None,
    verbose: int = 0,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[RandomForestClassifier, dict]:
    """
    Train and optionally validate a Random Forest classifier model using Sklearn.

    Various options and configurations for model performance evaluation are available. No validation,
    split to train and validation parts, and cross-validation can be chosen. If validation is performed,
    metric(s) to calculate can be defined and validation process configured (cross-validation method,
    number of folds, size of the split). Depending on the details of the validation process,
    the output metrics dictionary can be empty, one-dimensional or nested.

    For more information about Sklearn Random Forest classifier, read the documentation here:
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html.

    Args:
        X: Training data.
        y: Target labels.
        validation_method: Validation method to use. "split" divides data into two parts, "kfold_cv"
            performs k-fold cross-validation, "skfold_cv" performs stratified k-fold cross-validation,
            "loo_cv" performs leave-one-out cross-validation and "none" will not validate model at all
            (in this case, all X and y will be used solely for training).
        metrics: Metrics to use for scoring the model. Defaults to "accuracy".
        split_size: Fraction of the dataset to be used as validation data (rest is used for training).
            Used only when validation_method is "split". Defaults to 0.2.
        cv_folds: Number of folds used in cross-validation. Used only when validation_method is "kfold_cv"
            or "skfold_cv". Defaults to 5.
        n_estimators: The number of trees in the forest. Defaults to 100.
        criterion: The function to measure the quality of a split. Defaults to "gini".
        max_depth: The maximum depth of the tree. Values must be >= 1 or None, in which case nodes are
            expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
            Defaults to None.
        verbose: Specifies if modeling progress and performance should be printed. 0 doesn't print,
            values 1 or above will produce prints.
        random_state: Seed for random number generation. Defaults to None.
        **kwargs: Additional parameters for Sklearn's RandomForestClassifier.

    Returns:
        The trained RandomForestClassifier and metric scores as a dictionary.

    Raises:
        InvalidParameterValueException: If some of the numeric parameters are given invalid input values.
        NonMatchingParameterLengthsException: X and y have mismatching sizes.
    """
    if not n_estimators >= 1:
        raise InvalidParameterValueException("N-estimators must be at least 1.")
    if max_depth is not None and not max_depth >= 1:
        raise InvalidParameterValueException("Max depth must be at least 1 or None.")
    if verbose < 0:
        raise InvalidParameterValueException("Verbose must be a non-negative number.")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        random_state=random_state,
        verbose=verbose,
        **kwargs,
    )

    model, metrics = _train_and_validate_sklearn_model(
        X=X,
        y=y,
        model=model,
        validation_method=validation_method,
        metrics=metrics,
        split_size=split_size,
        cv_folds=cv_folds,
        random_state=random_state,
    )

    return model, metrics


@beartype
def random_forest_regressor_train(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    validation_method: Literal["split", "kfold_cv", "skfold_cv", "loo_cv", "none"] = "split",
    metrics: Sequence[Literal["mse", "rmse", "mae", "r2"]] = ["mse"],
    split_size: float = 0.2,
    cv_folds: int = 5,
    n_estimators: int = 100,
    criterion: Literal["squared_error", "absolute_error", "friedman_mse", "poisson"] = "squared_error",
    max_depth: Optional[int] = None,
    verbose: int = 0,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[RandomForestRegressor, dict]:
    """
    Train and optionally validate a Random Forest regressor model using Sklearn.

    Various options and configurations for model performance evaluation are available. No validation,
    split to train and validation parts, and cross-validation can be chosen. If validation is performed,
    metric(s) to calculate can be defined and validation process configured (cross-validation method,
    number of folds, size of the split). Depending on the details of the validation process,
    the output metrics dictionary can be empty, one-dimensional or nested.

    For more information about Sklearn Random Forest regressor, read the documentation here:
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html.

    Args:
        X: Training data.
        y: Target labels.
        validation_method: Validation method to use. "split" divides data into two parts, "kfold_cv"
            performs k-fold cross-validation, "skfold_cv" performs stratified k-fold cross-validation,
            "loo_cv" performs leave-one-out cross-validation and "none" will not validate model at all
            (in this case, all X and y will be used solely for training).
        metrics: Metrics to use for scoring the model. Defaults to "mse".
        split_size: Fraction of the dataset to be used as validation data (rest is used for training).
            Used only when validation_method is "split". Defaults to 0.2.
        cv_folds: Number of folds used in cross-validation. Used only when validation_method is "kfold_cv"
            or "skfold_cv". Defaults to 5.
        n_estimators: The number of trees in the forest. Defaults to 100.
        criterion: The function to measure the quality of a split. "absolute_error" results in significantly
            longer training time than "squared_error". Defaults to "squared_error".
        max_depth: The maximum depth of the tree. Values must be >= 1 or None, in which case nodes are
            expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
            Defaults to None.
        verbose: Specifies if modeling progress and performance should be printed. 0 doesn't print,
            values 1 or above will produce prints.
        random_state: Seed for random number generation. Defaults to None.
        **kwargs: Additional parameters for Sklearn's RandomForestRegressor.

    Returns:
        The trained RandomForestRegressor and metric scores as a dictionary.

    Raises:
        InvalidParameterValueException: If some of the numeric parameters are given invalid input values.
        NonMatchingParameterLengthsException: X and y have mismatching sizes.
    """
    if not n_estimators >= 1:
        raise InvalidParameterValueException("N-estimators must be at least 1.")
    if max_depth is not None and not max_depth >= 1:
        raise InvalidParameterValueException("Max depth must be at least 1 or None.")
    if verbose < 0:
        raise InvalidParameterValueException("Verbose must be a non-negative number.")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        random_state=random_state,
        verbose=verbose,
        **kwargs,
    )

    model, metrics = _train_and_validate_sklearn_model(
        X=X,
        y=y,
        model=model,
        validation_method=validation_method,
        metrics=metrics,
        split_size=split_size,
        cv_folds=cv_folds,
        random_state=random_state,
    )

    return model, metrics
