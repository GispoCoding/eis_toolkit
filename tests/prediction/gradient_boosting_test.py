import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from eis_toolkit import exceptions
from eis_toolkit.prediction.gradient_boosting import (
    gradient_boosting_classifier_predict,
    gradient_boosting_classifier_train,
    gradient_boosting_regressor_predict,
    gradient_boosting_regressor_train,
)

X, y = load_iris(return_X_y=True)


def test_gradient_boosting_classifier():
    """Test that Gradient Boosting classifier works as expected."""
    model, report_dict = gradient_boosting_classifier_train(X, y, random_state=42)
    predicted_labels = gradient_boosting_classifier_predict(model, X)

    assert isinstance(model, GradientBoostingClassifier)
    np.testing.assert_equal(len(predicted_labels), len(y))

    # Test that all predicted labels have perfect metric scores since we are predicting with the test data
    labels = ["0", "1", "2"]
    metrics = ["precision", "recall", "f1-score"]
    for label in labels:
        for metric in metrics:
            np.testing.assert_equal(report_dict[label][metric], 1.0)


def test_gradient_boosting_classifier_wrong_input_shapes():
    """Test that incorrectly shaped inputs raises the correct exception."""
    y_modified = y[:-1]
    with pytest.raises(exceptions.NonMatchingParameterLengthsException):
        gradient_boosting_classifier_train(X, y_modified, random_state=42)


def test_gradient_boosting_regressor():
    """Test that Gradient Boosting regressor works as expected."""
    model, report_dict = gradient_boosting_regressor_train(X, y, random_state=42)
    predicted_labels = gradient_boosting_regressor_predict(model, X)

    assert isinstance(model, GradientBoostingRegressor)
    np.testing.assert_equal(len(predicted_labels), len(y))

    np.testing.assert_almost_equal(report_dict["MAE"], 0.026911, decimal=4)
    np.testing.assert_almost_equal(report_dict["MSE"], 0.00350, decimal=4)
    np.testing.assert_almost_equal(report_dict["RMSE"], 0.05920, decimal=4)
    np.testing.assert_almost_equal(report_dict["R2"], 0.99502, decimal=4)


def test_gradient_boosting_regressor_wrong_input_shapes():
    """Test that incorrectly shaped inputs raises the correct exception."""
    y_modified = y[:-1]
    with pytest.raises(exceptions.NonMatchingParameterLengthsException):
        gradient_boosting_regressor_train(X, y_modified, random_state=42)
