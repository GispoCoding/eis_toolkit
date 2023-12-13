from numbers import Number
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Dict, List, Literal, Optional, Sequence, Tuple, Union
from scipy import sparse
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold, train_test_split
from tensorflow import keras

from eis_toolkit import exceptions

SPLIT = "split"
KFOLD_CV = "kfold_cv"
SKFOLD_CV = "skfold_cv"
LOO_CV = "loo_cv"
NO_VALIDATION = "none"


@beartype
def save_model(model: BaseEstimator, path: Path) -> None:
    """
    Save a trained Sklearn model to a .joblib file.

    Args:
        model: Trained model.
        path: Path where the model should be saved. Include the .joblib file extension.
    """
    joblib.dump(model, path)


@beartype
def load_model(path: Path) -> BaseEstimator:
    """
    Load a Sklearn model from a .joblib file.

    Args:
        path: Path from where the model should be loaded. Include the .joblib file extension.

    Returns:
        Loaded model.
    """
    return joblib.load(path)


@beartype
def split_data(
    *data: Union[np.ndarray, pd.DataFrame, sparse._csr.csr_matrix, List[Number]],
    split_size: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True,
) -> List[Union[np.ndarray, pd.DataFrame, sparse._csr.csr_matrix, List[Number]]]:
    """
    Split data into two parts. Can be used for train-test or train-validation splits.

    For more guidance, read documentation of sklearn.model_selection.train_test_split:
    (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).

    Args:
        *data: Data to be split. Multiple datasets can be given as input (for example X and y),
            but they need to have the same length. All datasets are split into two and the parts returned
            (for example X_train, X_test, y_train, y_test).
        split_size: The proportion of the second part of the split. Typically this is the size of test/validation
            part. The first part will be complemental proportion. For example, if split_size = 0.2, the first part
            will have 80% of the data and the second part 20% of the data. Defaults to 0.2.
        random_state: Seed for random number generation. Defaults to None.
        shuffle: If data is shuffled before splitting. Defaults to True.

    Returns:
        List containing splits of inputs (two outputs per input).
    """

    if not (0 < split_size < 1):
        raise exceptions.InvalidParameterValueException("Split size must be more than 0 and less than 1.")

    split_data = train_test_split(*data, test_size=split_size, random_state=random_state, shuffle=shuffle)

    return split_data


@beartype
def test_model(
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.Series],
    model: Union[BaseEstimator, keras.Model],
    metrics: Optional[Sequence[Literal["mse", "rmse", "mae", "r2", "accuracy", "precision", "recall", "f1"]]] = None,
) -> Dict[str, Number]:
    """
    Test and score a trained model.

    Args:
        X_test: Test data.
        y_test: Target labels for test data.
        model: Trained Sklearn classifier or regressor.
        metrics: Metrics to use for scoring the model. Defaults to "accuracy" for a classifier
            and to "mse" for a regressor.

    Returns:
        Test metric scores as a dictionary.
    """
    x_size = X_test.index.size if isinstance(X_test, pd.DataFrame) else X_test.shape[0]
    if x_size != y_test.size:
        raise exceptions.NonMatchingParameterLengthsException(
            f"X and y must have the length {x_size} != {y_test.size}."
        )

    if metrics is None:
        metrics = ["accuracy"] if is_classifier(model) else ["mse"]

    y_pred = model.predict(X_test)

    out_metrics = {}
    for metric in metrics:
        score = _score_model(model, y_test, y_pred, metric)
        out_metrics[metric] = score

    return out_metrics


@beartype
def predict(data: Union[np.ndarray, pd.DataFrame], model: Union[BaseEstimator, keras.Model]) -> np.ndarray:
    """
    Predict with a trained model.

    Args:
        data: Data used to make predictions.
        model: Trained classifier or regressor. Can be any machine learning model trained with
            EIS Toolkit (Sklearn and Keras models).

    Returns:
        Predictions.
    """
    result = model.predict(data)
    return result


@beartype
def _train_and_validate_sklearn_model(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    model: BaseEstimator,
    validation_method: Literal["split", "kfold_cv", "skfold_cv", "loo_cv", "none"],
    metrics: Sequence[Literal["mse", "rmse", "mae", "r2", "accuracy", "precision", "recall", "f1"]],
    split_size: float = 0.2,
    cv_folds: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> Tuple[BaseEstimator, dict]:
    """
    Train and validate Sklearn model.

    Serves as a common private/inner function for Random Forest, Logistic Regression and Gradient Boosting
    public functions.
    """

    # Perform checks
    x_size = X.index.size if isinstance(X, pd.DataFrame) else X.shape[0]
    if x_size != y.size:
        raise exceptions.NonMatchingParameterLengthsException(f"X and y must have the length {x_size} != {y.size}.")
    if len(metrics) == 0 and validation_method != NO_VALIDATION:
        raise exceptions.InvalidParameterValueException(
            "Metrics must have at least one chosen metric to validate model."
        )
    if cv_folds < 2:
        raise exceptions.InvalidParameterValueException("Number of cross-validation folds must be at least 2.")

    # Approach 1: No validation
    if validation_method == NO_VALIDATION:
        model.fit(X, y)
        metrics = {}

        return model, metrics

    # Approach 2: Validation with splitting data once
    elif validation_method == SPLIT:
        X_train, X_valid, y_train, y_valid = split_data(
            X, y, split_size=split_size, random_state=random_state, shuffle=shuffle
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)

        out_metrics = {}
        for metric in metrics:
            score = _score_model(model, y_valid, y_pred, metric)
            out_metrics[metric] = score

    # Approach 3: Cross-validation
    elif validation_method in [KFOLD_CV, SKFOLD_CV, LOO_CV]:
        cv = _get_cross_validator(validation_method, cv_folds, shuffle, random_state)

        # Initialize output metrics dictionary
        out_metrics = {}
        for metric in metrics:
            out_metrics[metric] = {}
            out_metrics[metric][f"{metric}_all"] = []

        # Loop over cross-validation folds and save metric scores
        for train_index, valid_index in cv.split(X, y):
            model.fit(X[train_index], y[train_index])
            y_pred = model.predict(X[valid_index])

            for metric in metrics:
                score = _score_model(model, y[valid_index], y_pred, metric)
                all_scores = out_metrics[metric][f"{metric}_all"]
                all_scores.append(score)

        # Calculate mean and standard deviation for all metrics
        for metric in metrics:
            scores = out_metrics[metric][f"{metric}_all"]
            out_metrics[metric][f"{metric}_mean"] = np.mean(scores)
            out_metrics[metric][f"{metric}_std"] = np.std(scores)

        # Fit on entire dataset after cross-validation
        model.fit(X, y)

        # If we calculated only 1 metric, remove the outer dictionary layer from output
        if len(out_metrics) == 1:
            out_metrics = out_metrics[metrics[0]]

    else:
        raise Exception(f"Unrecognized validation method: {validation_method}")

    return model, out_metrics


@beartype
def _score_model(
    model: BaseEstimator,
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    metric: Literal["mse", "rmse", "mae", "r2", "accuracy", "precision", "recall", "f1"],
) -> float:
    """Score a Sklearn model's predictions using the selected metric."""

    if metric in ["mae", "mse", "rmse", "r2"] and not is_regressor(model):
        raise exceptions.InvalidParameterValueException(
            f"Chosen metric ({metric}) is not applicable for given model type (classifier)."
        )
    if metric in ["accuracy", "precision", "recall", "f1"] and not is_classifier(model):
        raise exceptions.InvalidParameterValueException(
            f"Chosen metric ({metric}) is not applicable for given model type (regressor)."
        )

    if is_classifier(model):
        if len(y_true) > 2:  # Multiclass prediction
            average_method = "micro"
        else:  # Binary prediction
            average_method = "binary"

    if metric == "mae":
        score = mean_absolute_error(y_true, y_pred)
    elif metric == "mse":
        score = mean_squared_error(y_true, y_pred)
    elif metric == "rmse":
        score = mean_squared_error(y_true, y_pred, squared=False)
    elif metric == "r2":
        score = r2_score(y_true, y_pred)
    elif metric == "accuracy":
        score = accuracy_score(y_true, y_pred)
    elif metric == "precision":
        score = precision_score(y_true, y_pred, average=average_method)
    elif metric == "recall":
        score = recall_score(y_true, y_pred, average=average_method)
    elif metric == "f1":
        score = f1_score(y_true, y_pred, average=average_method)
    else:
        raise exceptions.InvalidParameterValueException(f"Unrecognized metric: {metric}")

    return score


@beartype
def _get_cross_validator(
    cv: str, folds: int, shuffle: bool, random_state: Optional[int]
) -> Union[KFold, StratifiedKFold, LeaveOneOut]:
    """Create and return a Sklearn cross-validator based on given parameter values."""
    if cv == KFOLD_CV:
        cross_validator = KFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
    elif cv == SKFOLD_CV:
        cross_validator = StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
    elif cv == LOO_CV:
        cross_validator = LeaveOneOut()
    else:
        raise exceptions.InvalidParameterValueException(f"CV method was not recognized: {cv}")

    return cross_validator
