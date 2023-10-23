from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from beartype import beartype
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from eis_toolkit import exceptions
from eis_toolkit.prediction.model_utils import evaluate_regression_model, tune_model_parameters


@beartype
def gradient_boosting_classifier_train(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    test_size: float = 0.25,
    loss: Literal["log_loss", "exponential"] = "log_loss",
    learning_rate: float = 0.1,
    n_estimators: int = 100,
    max_depth: int = 3,
    subsample: float = 1.0,
    random_state: Optional[int] = None,
    tune_with_method: Optional[Literal["grid", "random"]] = None,
    tune_parameters: Optional[dict] = None,
    tune_cv: int = 5,
    **kwargs,
) -> Tuple[GradientBoostingClassifier, dict]:
    """
    Train a Gradient Boosting classifier model using Sklearn with optional hyperparameter tuning.

    Trains the model with the given parameters and evaluates model performance using test data.

    Args:
        X: Training data.
        y: Target labels.
        test_size: Fraction of the dataset to be used as test data (rest is used for training). Defaults to 0.25.
        loss: The loss function to be optimized. Defaults to "log_loss" (same as in logistic regression).
        learning_rate: Shrinks the contribution of each tree. Values must be > 0. Defaults to 0.1.
        n_estimators: The number of boosting stages to run. Gradient boosting is fairly robust to over-fitting
            so a large number can result in better performance. Values must be >= 1. Defaults to 100.
        max_depth: Maximum depth of the individual regression estimators. The maximum depth limits the number
            of nodes in the tree.
        subsample: The fraction of samples to be used for fitting the individual base learners.
            If smaller than 1.0 this results in Stochastic Gradient Boosting. Subsample interacts with the
            parameter n_estimators. Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.
            Values must be in the range 0.0 < x <= 1.0. Defaults to 1.
        random_state: Seed for random number generation. Defaults to None.
        tune_with_method: If the model parameters should be tuned. Options include
            GridSearchCV and RandomizedSearchCV. Defaults to None, in which case
            model is not tuned.
        tune_parameters: Hyperparameters to be tuned (if model tuning is selected).
            Dictionary where keys parameter names (e.g. n_estimators) and values are lists
            of possible parameter values (e.g. [10, 50, 100]). Tune parameters must be defined
            is tuning is selected, otherwise this parameter is not used. Defaults to None.
        tune_cv: Number of cross-validation folds used in hyperparameter tuning. Defaults to 5.
        **kwargs: Additional parameters for Sklearn's GradientBoostingClassifier.

    Returns:
        The trained GradientBoostingClassifier and details of test set performance.

    Raises:
        NonMatchingParameterLengthsException: If length of X and y don't match.
    """
    x_size = X.index.size if isinstance(X, pd.DataFrame) else X.shape[0]
    if x_size != y.size:
        raise exceptions.NonMatchingParameterLengthsException(f"X and y must have the length {x_size} != {y.size}.")

    # Splitting data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = GradientBoostingClassifier(
        loss=loss,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        subsample=subsample,
        random_state=random_state,
        **kwargs,
    )

    # Training and optionally tuning the model
    if tune_with_method is not None:
        model = tune_model_parameters(model, tune_with_method, X_train, y_train, tune_parameters, cv=tune_cv)
    else:
        model.fit(X_train, y_train)

    # Predictions for test data
    y_pred = model.predict(X_test)

    # Performance metrics
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, report


@beartype
def gradient_boosting_classifier_predict(
    model: GradientBoostingClassifier, X: Union[np.ndarray, pd.DataFrame]
) -> np.ndarray:
    """
    Use a trained Gradient Boosting model to make predictions.

    Args:
        model: Trained GradientBoostingClassifier.
        X: Features for which predictions are to be made.

    Returns:
        Predicted labels.
    """
    predictions = model.predict(X)
    return predictions


@beartype
def gradient_boosting_regressor_train(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    test_size: float = 0.25,
    loss: Literal["squared_error", "absolute_error", "huber", "quantile"] = "squared_error",
    learning_rate: float = 0.1,
    n_estimators: int = 100,
    max_depth: int = 3,
    subsample: float = 1.0,
    random_state: Optional[int] = None,
    tune_with_method: Optional[Literal["grid", "random"]] = None,
    tune_parameters: Optional[dict] = None,
    tune_cv: int = 5,
    **kwargs,
) -> Tuple[GradientBoostingRegressor, dict]:
    """
    Train a Gradient Boosting regressor model using Sklearn with optional hyperparameter tuning.

    Trains the model with the given parameters and evaluates model performance using test data.

    Args:
        X: Training data.
        y: Target labels.
        test_size: Fraction of the dataset to be used as test data (rest is used for training). Defaults to 0.25.
        loss: The loss function to be optimized. Defaults to "squared_error".
        learning_rate: Shrinks the contribution of each tree. Values must be > 0. Defaults to 0.1.
        n_estimators: The number of boosting stages to run. Gradient boosting is fairly robust to over-fitting
            so a large number can result in better performance. Values must be >= 1. Defaults to 100.
        max_depth: Maximum depth of the individual regression estimators. The maximum depth limits the number
            of nodes in the tree.
        subsample: The fraction of samples to be used for fitting the individual base learners.
            If smaller than 1.0 this results in Stochastic Gradient Boosting. Subsample interacts with the
            parameter n_estimators. Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.
            Values must be in the range 0.0 < x <= 1.0. Defaults to 1.
        random_state: Seed for random number generation. Defaults to None.
        tune_with_method: If the model parameters should be tuned. Options include
            GridSearchCV and RandomizedSearchCV. Defaults to None, in which case
            model is not tuned.
        tune_parameters: Hyperparameters to be tuned (if model tuning is selected).
            Dictionary where keys parameter names (e.g. n_estimators) and values are lists
            of possible parameter values (e.g. [10, 50, 100]). Tune parameters must be defined
            is tuning is selected, otherwise this parameter is not used. Defaults to None.
        tune_cv: Number of cross-validation folds used in hyperparameter tuning. Defaults to 5.
        **kwargs: Additional parameters for Sklearn's GradientBoostingRegressor.

    Returns:
        The trained GradientBoostingRegressor and details of test set performance.

    Raises:
        NonMatchingParameterLengthsException: If length of X and y don't match.
    """
    x_size = X.index.size if isinstance(X, pd.DataFrame) else X.shape[0]
    if x_size != y.size:
        raise exceptions.NonMatchingParameterLengthsException(f"X and y must have the length {x_size} != {y.size}.")

    # Splitting data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = GradientBoostingRegressor(
        loss=loss,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        subsample=subsample,
        random_state=random_state,
        **kwargs,
    )

    # Training and optionally tuning the model
    if tune_with_method is not None:
        model = tune_model_parameters(model, tune_with_method, X_train, y_train, tune_parameters, cv=tune_cv)
    else:
        model.fit(X_train, y_train)

    # Predictions for test data
    y_pred = model.predict(X_test)

    # Performance metrics
    report = evaluate_regression_model(y_test, y_pred)

    return model, report


@beartype
def gradient_boosting_regressor_predict(
    model: GradientBoostingRegressor, X: Union[np.ndarray, pd.DataFrame]
) -> np.ndarray:
    """
    Use a trained Gradient Boosting regressor to make predictions.

    Args:
        model: Trained Gradient Boosting regressor.
        X: Features/data to predict.

    Returns:
        Predictions.
    """
    predictions = model.predict(X)
    return predictions
