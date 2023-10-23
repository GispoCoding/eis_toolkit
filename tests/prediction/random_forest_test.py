import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from eis_toolkit import exceptions
from eis_toolkit.prediction.random_forests import (
    random_forest_classifier_predict,
    random_forest_classifier_train,
    random_forest_regressor_predict,
    random_forest_regressor_train,
)

X, y = load_iris(return_X_y=True)


def test_random_forest_classifier():
    """Test that Random Forest classifier works as expected."""
    model, report_dict = random_forest_classifier_train(X, y, random_state=42)
    predicted_labels = random_forest_classifier_predict(model, X)

    assert isinstance(model, RandomForestClassifier)
    np.testing.assert_equal(len(predicted_labels), len(y))

    # Test that all predicted labels have perfect metric scores since we are predicting with the test data
    labels = ["0", "1", "2"]
    metrics = ["precision", "recall", "f1-score"]
    for label in labels:
        for metric in metrics:
            np.testing.assert_equal(report_dict[label][metric], 1.0)


def test_random_forest_classifier_wrong_input_shapes():
    """Test that incorrectly shaped inputs raises the correct exception."""
    y_modified = y[:-1]
    with pytest.raises(exceptions.NonMatchingParameterLengthsException):
        random_forest_classifier_train(X, y_modified, random_state=42)


def test_random_forest_regressor():
    """Test that Random Forest regressor works as expected."""
    model, report_dict = random_forest_regressor_train(X, y, random_state=42)
    predicted_labels = random_forest_regressor_predict(model, X)

    assert isinstance(model, RandomForestRegressor)
    np.testing.assert_equal(len(predicted_labels), len(y))

    np.testing.assert_almost_equal(report_dict["MAE"], 0.01105, decimal=4)
    np.testing.assert_almost_equal(report_dict["MSE"], 0.00086, decimal=4)
    np.testing.assert_almost_equal(report_dict["RMSE"], 0.02937, decimal=4)
    np.testing.assert_almost_equal(report_dict["R2"], 0.99877, decimal=4)


def test_random_forest_regressor_wrong_input_shapes():
    """Test that incorrectly shaped inputs raises the correct exception."""
    y_modified = y[:-1]
    with pytest.raises(exceptions.NonMatchingParameterLengthsException):
        random_forest_regressor_train(X, y_modified, random_state=42)
