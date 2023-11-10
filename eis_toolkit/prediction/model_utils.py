from pathlib import Path

import joblib
import numpy as np
from beartype import beartype
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@beartype
def save_model(model: BaseEstimator, filename: Path) -> None:
    """
    Save a trained sklearn model to a file.

    Args:
        model: Trained model.
        filename: Path where the model should be saved.
    """
    joblib.dump(model, filename)


@beartype
def load_model(filename: Path) -> BaseEstimator:
    """
    Load a sklearn model from a file.

    Args:
        filename: Path from where the model should be loaded.

    Returns:
        Loaded model.
    """
    return joblib.load(filename)


# NOTE: The implementation below is not used for now. However, it could be used in the future
# when/if parameter optimization is included

# @beartype
# def tune_model_parameters(
#     estimator: BaseEstimator,
#     method: Literal["grid", "random"],
#     X_train: Union[np.ndarray, pd.DataFrame],
#     y_train: Union[np.ndarray, pd.Series],
#     params: dict,
#     cv: int = 5,
# ) -> BaseEstimator:
#     """
#     Hyperparameter tuning using either GridSearch or RandomizedSearch.

#     Args:
#         estimator: The classifier to be tuned (e.g., RandomForestClassifier(), GradientBoostingClassifier()).
#         method: 'grid' for GridSearchCV, 'random' for RandomizedSearchCV.
#         X_train: Training data.
#         y_train: Target labels.
#         params: Hyperparameters to tune and their possible values.
#         cv: Number of cross-validation folds.

#     Returns:
#         Best model from the search.
#     """
#     if method == "grid":
#         search = GridSearchCV(estimator, params, cv=cv)
#     elif method == "random":
#         search = RandomizedSearchCV(estimator, params, cv=cv, n_iter=10)
#     else:
#         raise exceptions.InvalidParameterValueException("Method should be either 'grid' or 'random'.")

#     search.fit(X_train, y_train)
#     return search.best_estimator_


@beartype
def evaluate_regression_model(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluate a regression model and return various metrics.

    Produced metrics are:
        - MAE (mean absolute error)
        - MSE (mean square error),
        - RMSE (root mean square error)
        - R2 (R squared).

    Args:
        y_test: Test/true values.
        y_pred: Predicted values.

    Returns:
        Dictionary containing the computed metrics.
    """
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

    return metrics
