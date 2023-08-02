import numpy as np
import pandas as pd
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from eis_toolkit.exceptions import (
    EmptyArrayException,
    EmptyDataFrameException,
    InvalidArrayException,
    InvalidColumnException,
    InvalidParameterValueException,
)
from eis_toolkit.training.logistic_regression import logistic_regression

np.random.seed(0)
pl = np.round(np.random.uniform(4, 8, 10), 1)
pw = np.round(np.random.uniform(2, 4.5, 10), 1)
sl = np.round(np.random.uniform(1, 7, 10), 1)
sw = np.round(np.random.uniform(0.1, 3, 10), 1)
data = np.column_stack((pl, pw, sl, sw))
df = pd.DataFrame(data, columns=["pl", "pw", "sl", "sw"])
column_name = "species"

x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)


def test_empty_dataframe():
    """Test that empty DataFrame raises the correct exception."""
    with pytest.raises(EmptyDataFrameException):
        logistic_regression(pd.DataFrame(), column_name, x_train, x_test, y_train, y_test)


def test_empty_array():
    """Test that empty array raises the correct exception."""
    with pytest.raises(EmptyArrayException):
        logistic_regression(df, column_name, x_train, x_test, y_train, y_test=np.array([]))


def test_invalid_number_of_columns():
    """Test that invalid number of columns in input DataFrame raises the correct exception."""
    df = pd.DataFrame(np.zeros((1, 3)), columns=["pl", "pw", "sl"])
    with pytest.raises(InvalidColumnException):
        logistic_regression(df, column_name, x_train, x_test, y_train, y_test)


def test_invalid_array_shape():
    """Test that array with invalid shape raises the correct exception."""
    with pytest.raises(InvalidArrayException):
        logistic_regression(df, column_name, x_train, x_test, y_train, y_test=np.random.rand((5)))


def test_invalid_dataframe_column_dtype():
    """Test that invalid data type in any of DataFrame columns raises the correct exception."""
    invalid_df = pd.DataFrame({"pl": [0], "pw": [1], "sl": [2], "sw": ["foo"]})
    with pytest.raises(InvalidColumnException):
        logistic_regression(invalid_df, column_name, x_train, x_test, y_train, y_test)


def test_invalid_array_element_dtype():
    """Test that array element with invalid dtype raises the correct exception."""
    a = np.random.randint(3, size=37)
    b = np.array(["foo"])
    invalid_y_test = np.concatenate((a, b), axis=None)
    with pytest.raises(InvalidArrayException):
        logistic_regression(df, column_name, x_train, x_test, y_train, y_test=invalid_y_test)


def test_invalid_penalty():
    """Test that invalid input value for penalty raises the correct exception."""
    with pytest.raises(BeartypeCallHintParamViolation):
        logistic_regression(df, column_name, x_train, x_test, y_train, y_test, penalty="foo")


def test_invalid_max_iter():
    """Test that invalid input value for the maximum number of iterations raises the correct exception."""
    max_iter = 0
    with pytest.raises(InvalidParameterValueException):
        logistic_regression(df, column_name, x_train, x_test, y_train, y_test, max_iter=max_iter)


def test_logistic_regression_output():
    """Test that the logistic regression function labels data points correctly."""
    expected_labels = [2, 2, 1, 2, 0, 2, 1, 2, 1, 1]
    output = logistic_regression(df, column_name, x_train, x_test, y_train, y_test, max_iter=150)
    np.testing.assert_array_equal(output[column_name], expected_labels)
