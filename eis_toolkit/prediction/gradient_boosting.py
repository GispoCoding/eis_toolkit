from typing import Literal, Optional

import numpy as np
import pandas as pd
from beartype import beartype
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from eis_toolkit import exceptions
from eis_toolkit.prediction.model_utils import tune_model_parameters


@beartype
def gradient_boosting_classifier_train(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    n_estimators: int = 100,
    learning_rate: float = 1.0,
    random_state: Optional[int] = None,
    tune_with_method: Optional[Literal["grid", "random"]] = None,
    tune_parameters: Optional[dict] = None,
    **kwargs,
):
    """
    Train a Gradient Boosting model using Sklearn.

    Args:


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

    model = GradientBoostingClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state, **kwargs
    )

    # Tune model optionally
    if tune_with_method is not None:
        model = tune_model_parameters(model, tune_with_method, X_train, y_train, tune_parameters)
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
def gradient_boosting_classifier_predict(model: GradientBoostingClassifier, X: pd.DataFrame) -> pd.Series:
    """
    Use a trained Gradient Boosting model to make predictions.

    Args:
        model: Trained GradientBoostingClassifier.
        X: Features for which predictions are to be made.

    Returns:
        Predicted labels.
    """
    return model.predict(X)


@beartype
def gradient_boosting_regressor_train(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    n_estimators: int = 100,
    learning_rate: float = 1.0,
    random_state: Optional[int] = None,
    tune_with_method: Optional[Literal["grid", "random"]] = None,
    tune_parameters: Optional[dict] = None,
    cv: int = 5,
    **kwargs,
) -> GradientBoostingRegressor:
    """ """
    if X.shape[0] != y.shape[0]:
        raise exceptions.NonMatchingParameterLengthsException(
            f"X and y must have the length {X.shape[0]} != {X.shape[0]}."
        )

    # Splitting data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = GradientBoostingRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state, **kwargs
    )

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
def gradient_boosting_regressor_predict(model: GradientBoostingRegressor, X: np.ndarray) -> np.ndarray:
    """
    Use a trained Random Forest regressor to make predictions.

    Args:
        model: Trained random forest regressor.
        X: Features/data to predict.

    Returns:
        Predictions.
    """
    return model.predict(X)
