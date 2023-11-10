from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from beartype import beartype
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from eis_toolkit import exceptions
from eis_toolkit.prediction.model_utils import evaluate_regression_model


@beartype
def random_forest_classifier_train(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    test_size: float = 0.2,
    n_estimators: int = 100,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[RandomForestClassifier, dict]:
    """
    Train a Random Forest classifier model using Sklearn.

    Trains the model with the given parameters and evaluates model performance using test data.

    Args:
        X: Training data.
        y: Target labels.
        test_size: Fraction of the dataset to be used as test data (rest is used for training). Defaults to 0.2.
        n_estimators: The number of trees in the forest. Defaults to 100.
        random_state: Seed for random number generation. Defaults to None.
        **kwargs: Additional parameters for Sklearn's RandomForestClassifier.

    Returns:
        The trained RandomForestClassifier and details of test set performance.

    Raises:
        NonMatchingParameterLengthsException: If length of X and y don't match.
    """
    x_size = X.index.size if isinstance(X, pd.DataFrame) else X.shape[0]
    if x_size != y.size:
        raise exceptions.NonMatchingParameterLengthsException(f"X and y must have the length {x_size} != {y.size}.")

    # Splitting data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Creating the model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, **kwargs)

    # Training the model
    model.fit(X_train, y_train)

    # Predictions for test data
    y_pred = model.predict(X_test)

    # Performance metrics
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, report


@beartype
def random_forest_classifier_predict(model: RandomForestClassifier, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Use a trained Random Forest classifier model to make predictions.

    Args:
        model: Trained RandomForestClassifier.
        X: Features for which predictions are to be made.

    Returns:
        Predicted labels.
    """
    predictions = model.predict(X)
    return predictions


@beartype
def random_forest_regressor_train(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    test_size: float = 0.2,
    n_estimators: int = 100,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[RandomForestRegressor, dict]:
    """
    Train a Random Forest regressor model using Sklearn.

    Trains the model with the given parameters and evaluates model performance using test data.

    Args:
        X: Training data.
        y: Target labels.
        test_size: Fraction of the dataset to be used as test data (rest is used for training). Defaults to 0.2.
        n_estimators: The number of trees in the forest. Defaults to 100.
        random_state: Seed for random number generation. Defaults to None.
        **kwargs: Additional parameters for Sklearn's RandomForestRegressor.

    Returns:
        Trained Random Forest regressor.
    """
    x_size = X.index.size if isinstance(X, pd.DataFrame) else X.shape[0]
    if x_size != y.size:
        raise exceptions.NonMatchingParameterLengthsException(f"X and y must have the length {x_size} != {y.size}.")

    # Splitting data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, **kwargs)

    # Training the model
    model.fit(X_train, y_train)

    # Predictions for test data
    y_pred = model.predict(X_test)

    # Performance metrics
    report = evaluate_regression_model(y_test, y_pred)

    return model, report


@beartype
def random_forest_regressor_predict(model: RandomForestRegressor, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Use a trained Random Forest regressor to make predictions.

    Args:
        model: Trained Random Forest regressor.
        X: Features/data to predict.

    Returns:
        Predictions.
    """
    predictions = model.predict(X)
    return predictions
