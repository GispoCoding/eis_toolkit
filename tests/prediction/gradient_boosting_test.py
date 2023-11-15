import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from eis_toolkit import exceptions
from eis_toolkit.prediction.gradient_boosting import (
    gradient_boosting_classifier_train,
    gradient_boosting_regressor_train,
)

X_IRIS, Y_IRIS = load_iris(return_X_y=True)


def test_gradient_boosting_classifier():
    """Test that Gradient Boosting classifier works as expected."""
    metrics = ["accuracy"]
    model, out_metrics = gradient_boosting_classifier_train(X_IRIS, Y_IRIS, metrics=metrics, random_state=42)
    predicted_labels = model.predict(X_IRIS)

    assert isinstance(model, GradientBoostingClassifier)
    np.testing.assert_equal(len(predicted_labels), len(Y_IRIS))

    # Test that all predicted labels have perfect metric scores since we are predicting with the test data
    for metric in metrics:
        np.testing.assert_equal(out_metrics[metric], 1.0)


def test_gradient_boosting_regressor():
    """Test that Gradient Boosting regressor works as expected."""
    metrics = ["mae", "mse", "rmse", "r2"]
    model, out_metrics = gradient_boosting_regressor_train(X_IRIS, Y_IRIS, metrics=metrics, random_state=42)
    predicted_labels = model.predict(X_IRIS)

    assert isinstance(model, GradientBoostingRegressor)
    np.testing.assert_equal(len(predicted_labels), len(Y_IRIS))

    np.testing.assert_almost_equal(out_metrics["mae"], 0.03101, decimal=4)
    np.testing.assert_almost_equal(out_metrics["mse"], 0.00434, decimal=4)
    np.testing.assert_almost_equal(out_metrics["rmse"], 0.06593, decimal=4)
    np.testing.assert_almost_equal(out_metrics["r2"], 0.99377, decimal=4)


def test_invalid_learning_rate():
    """Test that invalid value for learning rate raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        gradient_boosting_classifier_train(X_IRIS, Y_IRIS, learning_rate=-1)
    with pytest.raises(exceptions.InvalidParameterValueException):
        gradient_boosting_regressor_train(X_IRIS, Y_IRIS, learning_rate=-1)


def test_invalid_n_estimators():
    """Test that invalid value for n estimators raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        gradient_boosting_classifier_train(X_IRIS, Y_IRIS, n_estimators=0)
    with pytest.raises(exceptions.InvalidParameterValueException):
        gradient_boosting_regressor_train(X_IRIS, Y_IRIS, n_estimators=0)


def test_invalid_max_depth():
    """Test that invalid value for max depth raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        gradient_boosting_classifier_train(X_IRIS, Y_IRIS, max_depth=0)
    with pytest.raises(exceptions.InvalidParameterValueException):
        gradient_boosting_regressor_train(X_IRIS, Y_IRIS, max_depth=0)


def test_invalid_subsample():
    """Test that invalid value for subsample raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        gradient_boosting_classifier_train(X_IRIS, Y_IRIS, subsample=0)
    with pytest.raises(exceptions.InvalidParameterValueException):
        gradient_boosting_regressor_train(X_IRIS, Y_IRIS, subsample=0)
