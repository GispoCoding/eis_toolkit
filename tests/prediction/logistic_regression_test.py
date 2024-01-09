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
    metrics = ["accuracy", "precision", "recall", "f1"]
    model, out_metrics = logistic_regression_train(X_IRIS, Y_IRIS, metrics=metrics, max_iter=150, random_state=42)
    predicted_labels = model.predict(X_IRIS)
    count_false = np.count_nonzero(predicted_labels - Y_IRIS)

    assert isinstance(model, LogisticRegression)
    np.testing.assert_equal(len(predicted_labels), len(Y_IRIS))

    np.testing.assert_equal(count_false, 3)
    np.testing.assert_equal(out_metrics["accuracy"], 1.0)
    np.testing.assert_equal(out_metrics["precision"], 1.0)
    np.testing.assert_equal(out_metrics["recall"], 1.0)
    np.testing.assert_equal(out_metrics["f1"], 1.0)


def test_invalid_penalty():
    """Test that invalid input value for penalty raises the correct exception."""
    with pytest.raises(BeartypeCallHintParamViolation):
        logistic_regression_train(X_IRIS, Y_IRIS, penalty="invalid_penalty")


def test_invalid_max_iter():
    """Test that invalid value for the maximum number of iterations raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        logistic_regression_train(X_IRIS, Y_IRIS, max_iter=0)
