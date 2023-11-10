import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from eis_toolkit import exceptions
from eis_toolkit.prediction.logistic_regression import logistic_regression_predict, logistic_regression_train

X, y = load_iris(return_X_y=True)


def test_logistic_regression():
    """Test that Logistic Regression works as expected."""
    model, report_dict = logistic_regression_train(X, y, random_state=42)
    predicted_labels = logistic_regression_predict(model, X)

    assert isinstance(model, LogisticRegression)
    np.testing.assert_equal(len(predicted_labels), len(y))

    # Test that all predicted labels have perfect metric scores since we are predicting with the test data
    labels = ["0", "1", "2"]
    metrics = ["precision", "recall", "f1-score"]
    for label in labels:
        for metric in metrics:
            np.testing.assert_equal(report_dict[label][metric], 1.0)


def test_logistic_regression_wrong_input_shapes():
    """Test that incorrectly shaped inputs raises the correct exception."""
    y_modified = y[:-1]
    with pytest.raises(exceptions.NonMatchingParameterLengthsException):
        logistic_regression_train(X, y_modified, random_state=42)


def test_invalid_penalty():
    """Test that invalid input value for penalty raises the correct exception."""
    with pytest.raises(BeartypeCallHintParamViolation):
        logistic_regression_train(X, y, penalty="invalid_penalty")


def test_invalid_max_iter():
    """Test that invalid input value for the maximum number of iterations raises the correct exception."""
    with pytest.raises(exceptions.InvalidParameterValueException):
        logistic_regression_train(X, y, max_iter=0)
