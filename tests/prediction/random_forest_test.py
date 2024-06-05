import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.prediction.random_forests import random_forest_classifier_train, random_forest_regressor_train

X_IRIS, Y_IRIS = load_iris(return_X_y=True)


def test_random_forest_classifier():
    """Test that Random Forest classifier works as expected."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    model, out_metrics = random_forest_classifier_train(X_IRIS, Y_IRIS, metrics=metrics, random_state=42)
    predicted_labels = model.predict(X_IRIS)
    count_false = np.count_nonzero(predicted_labels - Y_IRIS)

    assert isinstance(model, RandomForestClassifier)
    np.testing.assert_equal(len(predicted_labels), len(Y_IRIS))

    np.testing.assert_equal(count_false, 0)
    np.testing.assert_equal(out_metrics["accuracy"], 1.0)
    np.testing.assert_equal(out_metrics["precision"], 1.0)
    np.testing.assert_equal(out_metrics["recall"], 1.0)
    np.testing.assert_equal(out_metrics["f1"], 1.0)


def test_random_forest_regressor():
    """Test that Random Forest regressor works as expected."""
    metrics = ["mae", "mse", "rmse", "r2"]
    model, out_metrics = random_forest_regressor_train(X_IRIS, Y_IRIS, metrics=metrics, random_state=42)
    predicted_labels = model.predict(X_IRIS)
    count_false = np.count_nonzero(predicted_labels - Y_IRIS)

    assert isinstance(model, RandomForestRegressor)
    np.testing.assert_equal(len(predicted_labels), len(Y_IRIS))

    np.testing.assert_equal(count_false, 35)
    np.testing.assert_equal(out_metrics["mae"], 0.014)
    np.testing.assert_equal(out_metrics["mse"], 0.001)
    np.testing.assert_equal(out_metrics["rmse"], 0.037)
    np.testing.assert_equal(out_metrics["r2"], 0.998)


def test_random_forest_invalid_n_estimators():
    """Test that invalid value for n estimators raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        random_forest_classifier_train(X_IRIS, Y_IRIS, n_estimators=0)
    with pytest.raises(InvalidParameterValueException):
        random_forest_regressor_train(X_IRIS, Y_IRIS, n_estimators=0)
