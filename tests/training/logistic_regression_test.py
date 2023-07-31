import numpy as np
import pandas as pd
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from eis_toolkit.exceptions import (
    EmptyArrayException,
    EmptyDataFrameException,
    InvalidArrayShapeException,
    InvalidParameterValueException,
)
from eis_toolkit.training.logistic_regression import logistic_regression

np.random.seed(0)
empty_df = pd.DataFrame()
a = np.round(np.random.uniform(4, 8, 10), 1)
b = np.round(np.random.uniform(2, 4.5, 10), 1)
c = np.round(np.random.uniform(1, 7, 10), 1)
d = np.round(np.random.uniform(0.1, 3, 10), 1)
data = np.column_stack((a, b, c, d))
df = pd.DataFrame(data, columns=["A", "B", "C", "D"])
x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)


def test_logistic_regression_output():
    """Test that the logistic regression function labels data points correctly."""
    expected_labels = [2, 2, 1, 2, 0, 2, 1, 2, 1, 1]
    output = logistic_regression(df, x_train, x_test, y_train, y_test, max_iter=150)
    np.testing.assert_array_equal(output["label"], expected_labels)


def test_empty_array():
    """Test that empty array raises the correct exception."""
    with pytest.raises(EmptyArrayException):
        logistic_regression(df, x_train, x_test, y_train, y_test=np.array([]))


def test_invalid_shape():
    """Test that array with invalid shape raises the correct exception."""
    with pytest.raises(InvalidArrayShapeException):
        logistic_regression(df, x_train, x_test, y_train, y_test=np.random.rand((5)))


def test_empty_dataframe():
    """Test that empty DataFrame raises the correct exception."""
    with pytest.raises(EmptyDataFrameException):
        logistic_regression(empty_df, x_train, x_test, y_train, y_test)


def test_invalid_penalty():
    """Test that invalid input value for penalty raises the correct exception."""
    with pytest.raises(BeartypeCallHintParamViolation):
        logistic_regression(df, x_train, x_test, y_train, y_test, penalty="foo")


def test_invalid_max_iter():
    """Test that invalid input value for the maximum number of iterations raises the correct exception."""
    max_iter = 0
    with pytest.raises(InvalidParameterValueException):
        logistic_regression(df, x_train, x_test, y_train, y_test, max_iter=max_iter)
