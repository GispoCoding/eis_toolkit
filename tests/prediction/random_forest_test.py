import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from eis_toolkit import exceptions
from eis_toolkit.prediction.random_forests import random_forest_classifier_train, random_forest_regressor_train

X, y = load_iris(return_X_y=True)


def test_random_forest_classifier():
    """Test that Random Forest classifier works as expected."""
    metrics = ["accuracy"]
    model, out_metrics = random_forest_classifier_train(X, y, metrics=metrics, random_state=42)
    predicted_labels = model.predict(X)

    assert isinstance(model, RandomForestClassifier)
    np.testing.assert_equal(len(predicted_labels), len(y))

    # Test that all predicted labels have perfect metric scores since we are predicting with the test data
    for metric in out_metrics:
        np.testing.assert_equal(out_metrics[metric], 1.0)


def test_random_forest_regressor():
    """Test that Random Forest regressor works as expected."""
    metrics = ["mae", "mse", "rmse", "r2"]
    model, out_metrics = random_forest_regressor_train(X, y, metrics=metrics, random_state=42)
    predicted_labels = model.predict(X)

    assert isinstance(model, RandomForestRegressor)
    np.testing.assert_equal(len(predicted_labels), len(y))

    np.testing.assert_almost_equal(out_metrics["mae"], 0.01366, decimal=4)
    np.testing.assert_almost_equal(out_metrics["mse"], 0.00138, decimal=4)
    np.testing.assert_almost_equal(out_metrics["rmse"], 0.03719, decimal=4)
    np.testing.assert_almost_equal(out_metrics["r2"], 0.99802, decimal=4)


def test_random_forest_invalid_n_estimators():
    """Test that invalid value for n estimators raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        random_forest_classifier_train(X, y, n_estimators=0)
    with pytest.raises(exceptions.InvalidParameterValueException):
        random_forest_regressor_train(X, y, n_estimators=0)
