import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from eis_toolkit import exceptions
from eis_toolkit.prediction.logistic_regression import logistic_regression_train

X_IRIS, Y_IRIS = load_iris(return_X_y=True)


def test_logistic_regression():
    """Test that Logistic Regression works as expected."""
    metrics = ["accuracy"]
    model, out_metrics = logistic_regression_train(X_IRIS, Y_IRIS, random_state=42)
    predicted_labels = model.predict(X_IRIS)

    assert isinstance(model, LogisticRegression)
    np.testing.assert_equal(len(predicted_labels), len(Y_IRIS))

    # Test that all predicted labels have perfect metric scores since we are predicting with the test data
    for metric in metrics:
        np.testing.assert_equal(out_metrics[metric], 1.0)


def test_invalid_penalty():
    """Test that invalid input value for penalty raises the correct exception."""
    with pytest.raises(BeartypeCallHintParamViolation):
        logistic_regression_train(X_IRIS, Y_IRIS, penalty="invalid_penalty")


def test_invalid_max_iter():
    """Test that invalid value for the maximum number of iterations raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        logistic_regression_train(X_IRIS, Y_IRIS, max_iter=0)
