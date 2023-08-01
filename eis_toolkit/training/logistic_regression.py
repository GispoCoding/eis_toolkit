import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Literal
from sklearn.linear_model import LogisticRegression

from eis_toolkit.exceptions import (
    EmptyArrayException,
    EmptyDataFrameException,
    InvalidArrayException,
    InvalidColumnException,
    InvalidParameterValueException,
)


def _logistic_regression(
    data: pd.DataFrame,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    penalty: Literal,
    max_iter: int,
) -> pd.DataFrame:

    # Train the model
    model = _train_model(x_train, x_test, y_train, y_test, penalty, max_iter)

    # Make predictions on data using the trained model
    predictions = model.predict(np.array(data))

    # Add predicted labels into the DataFrame
    data["label"] = predictions

    return data


def _train_model(
    x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, penalty: Literal, max_iter: int
) -> LogisticRegression:
    model = LogisticRegression(penalty=penalty, max_iter=max_iter)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print(f"Test score: {score}")

    return model


@beartype
def logistic_regression(
    data: pd.DataFrame,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    penalty: Literal["l1", "l2", "elasicnet", None] = "l2",
    max_iter: int = 100,
) -> pd.DataFrame:
    """
    Perform logistic regression on input data.

    Args:
        data: Data for which predictions are made.
        x_train: Train samples.
        x_test: Test samples.
        y_train: Train labels.
        y_test: Test labels.
        penalty: Specifies the norm of the penalty. Defaults to 'l2'.
        max_iter: Maximum number of iterations taken to converge. Defaults to 100.

    Returns:
        DataFrame containing the assigned labels.

    Raises:
        EmptyDataFrameException: The input DataFrame is empty.
        EmptyArrayException: Any of the input arrays is empty.
        InvalidArrayException:  The corresponding sample and label arrays are of different lengths or there is at least
                                one non-numeric element in one of the input arrays.
        InvalidParameterValueException: The maximum number of iterations is not at least one.
        InvalidColumnException: The number of columns in DataFrame is not equal to the number of columns in training
                                sample array or there is at least one non-numeric column in the DataFrame.
    """
    if data.empty:
        raise EmptyDataFrameException("The input DataFrame is empty.")

    if any([x_train.size == 0, x_test.size == 0, y_train.size == 0, y_test.size == 0]):
        raise EmptyArrayException("All the input arrays must be non-empty.")

    if len(data.columns) != x_train.shape[1]:
        raise InvalidColumnException("The number of columns in DataFrame and training data array doesn't match.")

    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise InvalidArrayException("Corresponding sample and label arrays must have the same length.")

    if len(data.select_dtypes(include=np.number).columns) != len(data.columns):
        raise InvalidColumnException("All columns in DataFrame must be numeric.")

    if (
        not np.issubdtype(x_train.dtype, np.number)
        or not np.issubdtype(x_test.dtype, np.number)
        or not np.issubdtype(y_train.dtype, np.integer)
        or not np.issubdtype(y_test.dtype, np.integer)
    ):
        raise InvalidArrayException("All array elements must be numeric.")

    if max_iter <= 0:
        raise InvalidParameterValueException("The input value for maximum number of iterations must be at least one.")

    return _logistic_regression(data, x_train, x_test, y_train, y_test, penalty, max_iter)
