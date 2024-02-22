import os
from numbers import Number
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from beartype import beartype
from beartype.typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
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

from eis_toolkit.exceptions import InvalidParameterValueException, NonMatchingParameterLengthsException

# from eis_toolkit.utilities.checks.raster import check_raster_grids
from eis_toolkit.vector_processing.rasterize_vector import rasterize_vector

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
        raise InvalidParameterValueException("Split size must be more than 0 and less than 1.")

    split_data = train_test_split(*data, test_size=split_size, random_state=random_state, shuffle=shuffle)

    return split_data


@beartype
def evaluate_model(
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.Series],
    model: Union[BaseEstimator, keras.Model],
    metrics: Optional[Sequence[Literal["mse", "rmse", "mae", "r2", "accuracy", "precision", "recall", "f1"]]] = None,
) -> Tuple[np.ndarray, Dict[str, Number]]:
    """
    Evaluate/score a trained model with test data.

    Predicts with the given test data and evaluates the predictions.

    Args:
        X_test: Test data.
        y_test: Target labels for test data.
        model: Trained Sklearn classifier or regressor.
        metrics: Metrics to use for scoring the model. Defaults to "accuracy" for a classifier
            and to "mse" for a regressor.

    Returns:
        Predictions and metric scores as a dictionary.
    """
    x_size = X_test.index.size if isinstance(X_test, pd.DataFrame) else X_test.shape[0]
    if x_size != y_test.size:
        raise NonMatchingParameterLengthsException(f"X and y must have the length {x_size} != {y_test.size}.")

    if metrics is None:
        metrics = ["accuracy"] if is_classifier(model) else ["mse"]

    y_pred = model.predict(X_test)

    out_metrics = {}
    for metric in metrics:
        score = _score_model(model, y_test, y_pred, metric)
        out_metrics[metric] = score

    return y_pred, out_metrics


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
def reshape_predictions(
    predictions: np.ndarray, height: int, width: int, nodata_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Reshape 1D prediction ouputs into 2D Numpy array.

    The output is ready to be visualized and saved as a raster.

    Args:
        predictions: A 1D Numpy array with raw prediction data from `predict` function.
        height: Height of the output array
        width: Width of the output array
        nodata_mask: Nodata mask used to reconstruct original shape of data. This is the same mask
            applied to data before predicting to remove nodata. If any nodata was removed
            before predicting, this mask is required to reconstruct the original shape of data.
            Defaults to None.

    Returns:
        Predictions as a 2D Numpy array in the original array shape.
    """
    full_predictions_array = np.full(width * height, np.nan, dtype=predictions.dtype)
    if nodata_mask is not None:
        full_predictions_array[~nodata_mask.ravel()] = predictions
    predictions_reshaped = full_predictions_array.reshape((height, width))
    return predictions_reshaped


@beartype
def prepare_data_for_ml(
    training_raster_files: Sequence[Union[str, os.PathLike]],
    label_file: Optional[Union[str, os.PathLike]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], rasterio.profiles.Profile, Any]:
    """
    Prepare data ready for machine learning model training.

    Performs the following steps:
    - Read all bands of all training rasters into a stacked Numpy array
    - Read label data (and rasterize if a vector file is given)
    - Create a nodata mask using all training rasters and labels, and mask nodata cells out

    Args:
        training_raster_files: List of filepaths of training rasters. Files should only include
            raster that have the same grid properties and extent.
        label_file: Filepath to label (deposits) data. File can be either a vector file or raster file.
            If a vector file is provided, it will be rasterized into similar grid as training rasters. If
            a raster file is provided, it needs to have same grid properties and extent as training rasters.
            Optional parameter and can be omitted if preparing data for predicting. Defaults to None.

    Returns:
        Training data (X) in prepared shape, target labels (y) in prepared shape (if `label_file` was given),
            refrence raster metadata and nodata mask applied to X and y.
    """

    def _read_and_stack_training_raster(filepath: Union[str, os.PathLike]) -> Tuple[np.ndarray, dict]:
        """Read all bands of raster file with training data in a stack."""
        with rasterio.open(filepath) as src:
            raster_data = np.stack([src.read(i) for i in range(1, src.count + 1)])
            profile = src.profile
        return raster_data, profile

    # Read and stack training rasters
    training_data, profiles = zip(*[_read_and_stack_training_raster(file) for file in training_raster_files])
    # # TODO. Waiting for check_raster_grids input modification to profiles
    # if not check_raster_grids(profiles, same_extent=True):
    #     raise NonMatchingRasterGridException
    reference_profile = profiles[0]
    nodata_values = [profile["nodata"] for profile in profiles]

    # Reshape training data for ML and create mask
    reshaped_data = []
    nodata_mask = None

    for raster, nodata in zip(training_data, nodata_values):
        raster_reshaped = raster.reshape(raster.shape[0], -1).T
        reshaped_data.append(raster_reshaped)

        if nodata is not None:
            raster_mask = (raster_reshaped == nodata).any(axis=1)
            nodata_mask = raster_mask if nodata_mask is None else nodata_mask | raster_mask

    X = np.concatenate(reshaped_data, axis=1)

    if label_file is not None:
        # Check label file type and process accordingly
        file_extension = os.path.splitext(label_file)[1].lower()

        # Labels/deposits in vector format
        if file_extension in [".shp", ".geojson", ".json", ".gpkg"]:
            y, _ = rasterize_vector(geodataframe=gpd.read_file(label_file), base_raster_profile=reference_profile)

        # Labels/deposits in raster format
        else:
            with rasterio.open(label_file) as label_raster:
                y = label_raster.read(1)  # Assuming labels are in the first band
                label_nodata = label_raster.nodata

            label_nodata_mask = y == label_nodata

            # Combine masks and apply to feature and label data
            nodata_mask = nodata_mask | label_nodata_mask.ravel()

        y = y.ravel()[~nodata_mask]

    else:
        y = None

    X = X[~nodata_mask]

    return X, y, reference_profile, nodata_mask


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

    Args:
        X: Training data.
        y: Target labels.
        model: Initialized, to-be-trained Sklearn model.
        validation_method: Validation method to use.
        metrics: Metrics to use for scoring the model.
        split_size: Fraction of the dataset to be used as validation data (for validation method "split").
            Defaults to 0.2.
        cv_folds: Number of folds used in cross-validation. Defaults to 5.
        shuffle: If data is shuffled before splitting. Defaults to True.
        random_state: Seed for random number generation. Defaults to None.

    Returns:
        Trained Sklearn model and metric scores as a dictionary.

    Raises:
        NonMatchingParameterLengthsException: X and y have mismatching sizes.
        InvalidParameterValueException: Validation method was chosen without any metric or `cv_folds` is too small.
    """
    # Perform checks
    x_size = X.index.size if isinstance(X, pd.DataFrame) else X.shape[0]
    if x_size != y.size:
        raise NonMatchingParameterLengthsException(f"X and y must have the length {x_size} != {y.size}.")
    if len(metrics) == 0 and validation_method != NO_VALIDATION:
        raise InvalidParameterValueException("Metrics must have at least one chosen metric to validate model.")
    if cv_folds < 2:
        raise InvalidParameterValueException("Number of cross-validation folds must be at least 2.")

    # Validation approach 1: No validation
    if validation_method == NO_VALIDATION:
        model.fit(X, y)
        metrics = {}

        return model, metrics

    # Validation approach 2: Validation with splitting data once
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

    # Validation approach 3: Cross-validation
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
        raise InvalidParameterValueException(f"Unrecognized validation method: {validation_method}")

    return model, out_metrics


@beartype
def _score_model(
    model: BaseEstimator,
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    metric: Literal["mse", "rmse", "mae", "r2", "accuracy", "precision", "recall", "f1"],
) -> float:
    """
    Score predictions of Sklearn model using the selected metric.

    Args:
        model: Sklearn model used to produce the predictions. Used only to identify the type
            of model (regressor or classifier).
        y_true: Target values ("ground truth") against which scoring is performed.
        y_pred: Predicted values generated by the model.
        metric: The scoring metric. Different options for regressor and classifier.

    Returns:
        The prediction score.

    Raises:
        InvalidParameterValueException: If model type is invalid for the given metric or metric was not recognized.
    """
    if metric in ["mae", "mse", "rmse", "r2"] and not is_regressor(model):
        raise InvalidParameterValueException(
            f"Chosen metric ({metric}) is not applicable for given model type (classifier)."
        )
    if metric in ["accuracy", "precision", "recall", "f1"] and not is_classifier(model):
        raise InvalidParameterValueException(
            f"Chosen metric ({metric}) is not applicable for given model type (regressor)."
        )

    if is_classifier(model):
        # Multiclass classification
        if len(y_true) > 2:
            average_method = "micro"
        # Binary classification
        else:
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
        raise InvalidParameterValueException(f"Unrecognized metric: {metric}")

    return score


@beartype
def _get_cross_validator(
    cv: Literal["kfold_cv", "skfold_cv", "loo_cv"], folds: int, shuffle: bool, random_state: Optional[int]
) -> Union[KFold, StratifiedKFold, LeaveOneOut]:
    """
    Create a Sklearn cross-validator.

    Args:
        cv: Name/identifier of the cross-validator.
        folds: Number of folds to use (for Kfold and StratifiedKFold).
        shuffle: If data is shuffled before splitting.
        random_state: Seed for random number generation.

    Returns:
        Sklearn cross-validator instance.

    Raises:
        InvalidParameterValueException: Invalid input for `cv`.
    """
    if cv == KFOLD_CV:
        cross_validator = KFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
    elif cv == SKFOLD_CV:
        cross_validator = StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
    elif cv == LOO_CV:
        cross_validator = LeaveOneOut()
    else:
        raise InvalidParameterValueException(f"CV method was not recognized: {cv}")

    return cross_validator
