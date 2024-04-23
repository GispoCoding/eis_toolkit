from os.path import exists
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from eis_toolkit.evaluation.scoring import score_predictions
from eis_toolkit.exceptions import InvalidParameterValueException, NonMatchingParameterLengthsException
from eis_toolkit.prediction.machine_learning_general import (
    _train_and_validate_sklearn_model,
    load_model,
    save_model,
    split_data,
)
from eis_toolkit.prediction.machine_learning_predict import predict_classifier

TEST_DIR = Path(__file__).parent.parent

X_IRIS, Y_IRIS = load_iris(return_X_y=True)

RF_MODEL = RandomForestClassifier()
CLF_METRICS = ["accuracy", "precision", "recall", "f1"]
REGR_METRICS = ["mse", "rmse", "mae", "r2"]


# NOTE: Testing loo_cv has been left out since it takes a lot longer than the other cv methods


def test_train_and_evaluate_with_no_validation():
    """Test that training a model without evaluation works as expected."""
    model, out_metrics = _train_and_validate_sklearn_model(
        X_IRIS, Y_IRIS, model=RF_MODEL, validation_method="none", metrics=CLF_METRICS, random_state=42
    )

    assert isinstance(model, RandomForestClassifier)
    assert not out_metrics


def test_train_and_evaluate_with_split():
    """Test that training a model with split validation works as expected."""
    model, out_metrics = _train_and_validate_sklearn_model(
        X_IRIS,
        Y_IRIS,
        model=RF_MODEL,
        validation_method="split",
        metrics=CLF_METRICS,
        split_size=0.25,
        random_state=42,
    )

    assert isinstance(model, RandomForestClassifier)
    np.testing.assert_equal(len(out_metrics), 4)


def test_train_and_evaluate_with_kfold_cv():
    """Test that training a model with k-fold cross-validation works as expected."""
    model, out_metrics = _train_and_validate_sklearn_model(
        X_IRIS, Y_IRIS, model=RF_MODEL, validation_method="kfold_cv", metrics=CLF_METRICS, cv_folds=3, random_state=42
    )

    assert isinstance(model, RandomForestClassifier)
    np.testing.assert_equal(len(out_metrics), 4)


def test_train_and_evaluate_with_skfold_cv():
    """Test that training a model with stratified k-fold cross-validation works as expected."""
    model, out_metrics = _train_and_validate_sklearn_model(
        X_IRIS, Y_IRIS, model=RF_MODEL, validation_method="skfold_cv", metrics=CLF_METRICS, cv_folds=3, random_state=42
    )

    assert isinstance(model, RandomForestClassifier)
    np.testing.assert_equal(len(out_metrics), 4)


def test_binary_classification():
    """Test that training with binary data works as expected."""
    X_binary = np.array(
        [
            [1.0, 1.1, 2.0, 1.5],
            [2.0, 2.4, 1.5, 1.4],
            [1.2, 1.5, 2.2, 1.7],
            [1.1, 0.9, 1.7, 1.4],
            [2.5, 2.1, 1.4, 1.1],
        ]
    )
    y_binary = np.array([1, 0, 1, 1, 0])

    model, out_metrics = _train_and_validate_sklearn_model(
        X_binary,
        y_binary,
        model=RF_MODEL,
        validation_method="kfold_cv",
        metrics=CLF_METRICS,
        cv_folds=3,
        random_state=42,
    )

    assert isinstance(model, RandomForestClassifier)
    np.testing.assert_equal(len(out_metrics), 4)


def test_splitting():
    """Test that split data works as expected."""
    X_train, X_test, y_train, y_test = split_data(X_IRIS, Y_IRIS, split_size=0.2, random_state=42)
    np.testing.assert_equal(len(X_train), len(X_IRIS) * 0.8)
    np.testing.assert_equal(len(y_train), len(Y_IRIS) * 0.8)
    np.testing.assert_equal(len(X_test), len(X_IRIS) * 0.2)
    np.testing.assert_equal(len(y_test), len(Y_IRIS) * 0.2)


def test_evaluate_model_sklearn():
    """Test that evaluating model works as expected with a Sklearn model."""
    X_train, X_test, y_train, y_test = split_data(X_IRIS, Y_IRIS, split_size=0.2, random_state=42)

    model, _ = _train_and_validate_sklearn_model(
        X_train, y_train, model=RF_MODEL, validation_method="none", metrics=CLF_METRICS, random_state=42
    )

    predictions = predict_classifier(X_test, model, include_probabilities=False)
    accuracy = score_predictions(y_test, predictions, "accuracy")
    np.testing.assert_equal(accuracy, 1.0)


def test_predict_classifier_sklearn():
    """Test that predicting with classifier works as expected with a Sklearn model."""
    X_train, X_test, y_train, y_test = split_data(X_IRIS, Y_IRIS, split_size=0.2, random_state=42)

    model, _ = _train_and_validate_sklearn_model(
        X_train, y_train, model=RF_MODEL, validation_method="none", metrics=CLF_METRICS, random_state=42
    )

    predicted_labels, predicted_probabilities = predict_classifier(X_test, model, True)
    np.testing.assert_equal(len(predicted_labels), len(y_test))
    np.testing.assert_equal(len(predicted_probabilities), len(y_test))


def test_save_and_load_model():
    """Test that saving and loading a model works as expected."""
    model_save_path = TEST_DIR.joinpath("data/local/results/saved_rf_model.joblib")

    save_model(RF_MODEL, model_save_path)
    assert exists(model_save_path)
    loaded_rf_model = load_model(model_save_path)
    assert isinstance(loaded_rf_model, RandomForestClassifier)


def test_mismatching_X_and_y():
    """Test that invalid lengths for X and y raises the correct exception."""
    with pytest.raises(NonMatchingParameterLengthsException):
        _train_and_validate_sklearn_model(
            X_IRIS, Y_IRIS[:-1], model=RF_MODEL, validation_method="none", metrics=CLF_METRICS
        )


def test_invalid_metrics():
    """Test that invalid metric selection raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        _train_and_validate_sklearn_model(X_IRIS, Y_IRIS, model=RF_MODEL, validation_method="split", metrics=[])


def test_invalid_cv_folds():
    """Test that invalid metric selection raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        _train_and_validate_sklearn_model(
            X_IRIS, Y_IRIS, model=RF_MODEL, validation_method="kfold_cv", metrics=CLF_METRICS, cv_folds=1
        )


def test_invalid_split_size():
    """Test that invalid metric selection raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        _train_and_validate_sklearn_model(
            X_IRIS, Y_IRIS, model=RF_MODEL, validation_method="split", metrics=CLF_METRICS, split_size=0.0
        )
