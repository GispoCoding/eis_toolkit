from typing import Literal, Optional

import numpy as np
import pandas as pd
from beartype import beartype
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from eis_toolkit import exceptions
from eis_toolkit.prediction.model_utils import tune_model_parameters


@beartype
def random_forest_classifier_train(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    n_estimators: int = 100,
    random_state: Optional[int] = None,
    tune_with_method: Optional[Literal["grid", "random"]] = None,
    tune_parameters: Optional[dict] = None,
    cv: int = 5,
    **kwargs,
) -> RandomForestClassifier:
    """
    Train a Random Forest model using Sklearn.

    Args:
        X: Training data.
        y: Target labels.
        test_size: Fraction of the dataset to be used as test data. Defaults to 0.25.
        n_estimators: The number of trees in the forest. Defaults to 100.
        random_state: Seed for random number generation. Defaults to None.
        tune_with_method: If the model parameters should be tuned. Options include
            GridSearchCV and RandomizedSearchCV. Defaults to None, in which case
            model is not tuned.
        tune_parameters: Hyperparameters to be tuned (if model tuning is selected).
            Dictionary where keys parameter names (e.g. n_estimators) and values are lists
            of possible parameter values (e.g. [10, 50, 100]). Tune parameters must be defined
            is tuning is selected, otherwise this parameter is not used. Defaults to None.
        cv: Number of cross-validation folds used in hyperparameter tuning. Defaults to 5.
        **kwargs: Additional parameters for RandomForestClassifier.

    Returns:
        The trained RandomForestClassifier and details of test set performance.

    Raises:
        NonMatchingParameterLengthsException: If length of X and y don't match.
    """
    if X.index.size != y.size:
        raise exceptions.NonMatchingParameterLengthsException(
            f"X and y must have the length {X.index.size} != {y.size}."
        )
    # Splitting data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Creating the model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, **kwargs)

    # Training and optionally tuning the model
    if tune_with_method is not None:
        model = tune_model_parameters(model, tune_with_method, X_train, y_train, tune_parameters, cv=cv)
    else:
        model.fit(X_train, y_train)

    # Getting predictions for the test set
    y_pred = model.predict(X_test)

    # Getting performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    performance = f"Accuracy: {accuracy}\n{report}"

    return model, performance


@beartype
def random_forest_classifier_predict(model: RandomForestClassifier, X: pd.DataFrame) -> pd.Series:
    """
    Use a trained Random Forest model to make predictions.

    Args:
        model: Trained RandomForestClassifier.
        X: Features for which predictions are to be made.

    Returns:
        Predicted labels.
    """
    return model.predict(X)


@beartype
def random_forest_regressor_train(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    n_estimators: int = 100,
    random_state: Optional[int] = None,
    tune_with_method: Optional[Literal["grid", "random"]] = None,
    tune_parameters: Optional[dict] = None,
    cv: int = 5,
    **kwargs,
) -> RandomForestRegressor:
    """
    Train a random forest regressor with optional hyperparameter tuning.

    Args:
        X: Training data.
        y: Target labels.
        test_size: Fraction of the dataset to be used as test data. Defaults to 0.25.
        n_estimators: The number of trees in the forest. Defaults to 100.
        random_state: Seed for random number generation. Defaults to None.
        tune_with_method: If the model parameters should be tuned. Options include
            GridSearchCV and RandomizedSearchCV. Defaults to None, in which case
            model is not tuned.
        tune_parameters: Hyperparameters to be tuned (if model tuning is selected).
            Dictionary where keys parameter names (e.g. n_estimators) and values are lists
            of possible parameter values (e.g. [10, 50, 100]). Tune parameters must be defined
            is tuning is selected, otherwise this parameter is not used. Defaults to None.
        cv: Number of cross-validation folds used in hyperparameter tuning. Defaults to 5.
        **kwargs: Additional parameters for RandomForestRegressor.

    Returns:
        Trained random forest regressor.
    """
    if X.shape[0] != y.shape[0]:
        raise exceptions.NonMatchingParameterLengthsException(
            f"X and y must have the length {X.shape[0]} != {X.shape[0]}."
        )

    # Splitting data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, **kwargs)

    # Training and optionally tuning the model
    if tune_with_method is not None:
        model = tune_model_parameters(model, tune_with_method, X_train, y_train, tune_parameters, cv=cv)
    else:
        model.fit(X_train, y_train)

    # Getting predictions for the test set
    y_pred = model.predict(X_test)

    # Getting performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    performance = f"Accuracy: {accuracy}\n{report}"

    return model, performance


@beartype
def random_forest_regressor_predict(model: RandomForestRegressor, X: np.ndarray) -> np.ndarray:
    """
    Use a trained random forest regressor to make predictions.

    Args:
        model: Trained random forest regressor.
        X: Features/data to predict.

    Returns:
        Predictions.
    """
    return model.predict(X)
