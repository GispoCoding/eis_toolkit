from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from eis_toolkit import exceptions


def save_model(model: BaseEstimator, filename: Path) -> None:
    """
    Save a trained sklearn model to a file.

    Args:
        model: Trained model.
        filename: Path where the model should be saved.
    """
    joblib.dump(model, filename)


def load_model(filename: Path) -> BaseEstimator:
    """
    Load a sklearn model from a file.

    Args:
        filename: Path from where the model should be loaded.

    Returns:
        Loaded model.
    """
    return joblib.load(filename)


def tune_model_parameters(
    estimator: BaseEstimator,
    method: Literal["grid", "random"],
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict,
    cv: int = 5,
) -> BaseEstimator:
    """
    Hyperparameter tuning using either GridSearch or RandomizedSearch.

    Args:
        estimator: The classifier to be tuned (e.g., RandomForestClassifier(), GradientBoostingClassifier()).
        method: 'grid' for GridSearchCV, 'random' for RandomizedSearchCV.
        X_train: Training data.
        y_train: Training data.
        params: Hyperparameters to tune and their possible values.
        cv: Number of cross-validation folds.

    Returns:
        Best model from the search.
    """

    if method == "grid":
        search = GridSearchCV(estimator, params, cv=cv)
    elif method == "random":
        search = RandomizedSearchCV(estimator, params, cv=cv, n_iter=10)
    else:
        raise exceptions.InvalidParameterValueException("Method should be either 'grid' or 'random'.")

    search.fit(X_train, y_train)
    return search.best_estimator_
