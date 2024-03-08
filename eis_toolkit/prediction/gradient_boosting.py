from numbers import Number

import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Literal, Optional, Sequence, Tuple, Union
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.prediction.machine_learning_general import _train_and_validate_sklearn_model


@beartype
def gradient_boosting_classifier_train(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    validation_method: Literal["split", "kfold_cv", "skfold_cv", "loo_cv", "none"] = "split",
    metrics: Sequence[Literal["accuracy", "precision", "recall", "f1", "auc"]] = ["accuracy"],
    split_size: float = 0.2,
    cv_folds: int = 5,
    loss: Literal["log_loss", "exponential"] = "log_loss",
    learning_rate: Number = 0.1,
    n_estimators: int = 100,
    max_depth: Optional[int] = 3,
    subsample: Number = 1.0,
    verbose: int = 0,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[GradientBoostingClassifier, dict]:
    """
    Train and optionally validate a Gradient Boosting classifier model using Sklearn.

    Various options and configurations for model performance evaluation are available. No validation,
    split to train and validation parts, and cross-validation can be chosen. If validation is performed,
    metric(s) to calculate can be defined and validation process configured (cross-validation method,
    number of folds, size of the split). Depending on the details of the validation process,
    the output metrics dictionary can be empty, one-dimensional or nested.

    For more information about Sklearn Gradient Boosting classifier read the documentation here:
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html.

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
        loss: The loss function to be optimized. Defaults to "log_loss" (same as in logistic regression).
        learning_rate: Shrinks the contribution of each tree. Values must be >= 0. Defaults to 0.1.
        n_estimators: The number of boosting stages to run. Gradient boosting is fairly robust to over-fitting
            so a large number can result in better performance. Values must be >= 1. Defaults to 100.
        max_depth: Maximum depth of the individual regression estimators. The maximum depth limits the number
            of nodes in the tree. Values must be >= 1 or None, in which case nodes are expanded until all leaves
            are pure or until all leaves contain less than min_samples_split samples. Defaults to 3.
        subsample: The fraction of samples to be used for fitting the individual base learners.
            If smaller than 1.0 this results in Stochastic Gradient Boosting. Subsample interacts with the
            parameter n_estimators. Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.
            Values must be in the range 0.0 < x <= 1.0. Defaults to 1.0.
        verbose: Specifies if modeling progress and performance should be printed. 0 doesn't print,
            1 prints once in a while depending on the number of tress, 2 or above will print for every tree.
        random_state: Seed for random number generation. Defaults to None.
        **kwargs: Additional parameters for Sklearn's GradientBoostingClassifier.

    Returns:
        The trained GradientBoostingClassifier and metric scores as a dictionary.

    Raises:
        InvalidParameterValueException: If some of the numeric parameters are given invalid input values.
        NonMatchingParameterLengthsException: X and y have mismatching sizes.
    """
    if not learning_rate >= 0:
        raise InvalidParameterValueException("Learning rate must be non-negative.")
    if not n_estimators >= 1:
        raise InvalidParameterValueException("N-estimators must be at least 1.")
    if max_depth is not None and not max_depth >= 1:
        raise InvalidParameterValueException("Max depth must be at least 1 or None.")
    if not (0 < subsample <= 1):
        raise InvalidParameterValueException("Subsample must be more than 0 and at most 1.")
    if verbose < 0:
        raise InvalidParameterValueException("Verbose must be a non-negative number.")

    model = GradientBoostingClassifier(
        loss=loss,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        subsample=subsample,
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
def gradient_boosting_regressor_train(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    validation_method: Literal["split", "kfold_cv", "skfold_cv", "loo_cv", "none"] = "split",
    metrics: Sequence[Literal["mse", "rmse", "mae", "r2"]] = ["mse"],
    split_size: float = 0.2,
    cv_folds: int = 5,
    loss: Literal["squared_error", "absolute_error", "huber", "quantile"] = "squared_error",
    learning_rate: Number = 0.1,
    n_estimators: int = 100,
    max_depth: Optional[int] = 3,
    subsample: Number = 1.0,
    verbose: int = 0,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[GradientBoostingRegressor, dict]:
    """
    Train and optionally validate a Gradient Boosting regressor model using Sklearn.

    Various options and configurations for model performance evaluation are available. No validation,
    split to train and validation parts, and cross-validation can be chosen. If validation is performed,
    metric(s) to calculate can be defined and validation process configured (cross-validation method,
    number of folds, size of the split). Depending on the details of the validation process,
    the output metrics dictionary can be empty, one-dimensional or nested.

    For more information about Sklearn Gradient Boosting regressor read the documentation here:
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html.

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
        loss: The loss function to be optimized. Defaults to "squared_error".
        learning_rate: Shrinks the contribution of each tree. Values must be > 0. Defaults to 0.1.
        n_estimators: The number of boosting stages to run. Gradient boosting is fairly robust to over-fitting
            so a large number can result in better performance. Values must be >= 1. Defaults to 100.
        max_depth: Maximum depth of the individual regression estimators. The maximum depth limits the number
            of nodes in the tree. Values must be >= 1 or None, in which case nodes are expanded until all leaves
            are pure or until all leaves contain less than min_samples_split samples. Defaults to 3.
        subsample: The fraction of samples to be used for fitting the individual base learners.
            If smaller than 1.0 this results in Stochastic Gradient Boosting. Subsample interacts with the
            parameter n_estimators. Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.
            Values must be in the range 0.0 < x <= 1.0. Defaults to 1.
        verbose: Specifies if modeling progress and performance should be printed. 0 doesn't print,
            1 prints once in a while depending on the number of tress, 2 or above will print for every tree.
        random_state: Seed for random number generation. Defaults to None.
        **kwargs: Additional parameters for Sklearn's GradientBoostingRegressor.

    Returns:
        The trained GradientBoostingRegressor and metric scores as a dictionary.

    Raises:
        InvalidParameterValueException: If some of the numeric parameters are given invalid input values.
        NonMatchingParameterLengthsException: X and y have mismatching sizes.
    """
    if not learning_rate >= 0:
        raise InvalidParameterValueException("Learning rate must be non-negative.")
    if not n_estimators >= 1:
        raise InvalidParameterValueException("N-estimators must be at least 1.")
    if max_depth is not None and not max_depth >= 1:
        raise InvalidParameterValueException("Max depth must be at least 1 or None.")
    if not (0 < subsample <= 1):
        raise InvalidParameterValueException("Subsample must be more than 0 and at most 1.")
    if verbose < 0:
        raise InvalidParameterValueException("Verbose must be a non-negative number.")

    model = GradientBoostingRegressor(
        loss=loss,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        subsample=subsample,
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
