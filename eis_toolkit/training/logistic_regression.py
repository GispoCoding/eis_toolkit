import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Literal
from sklearn.linear_model import LogisticRegression

from eis_toolkit.exceptions import (
    EmptyArrayException,
    EmptyDataFrameException,
    InvalidArrayShapeException,
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
        EmptyDataFrameException: The input data or training data DataFrame is empty.
        EmptyArrayException: Any of the input arrays is empty.
        InvalidArrayShapeException: The corresponding sample and label arrays are of different lengths.
        InvalidParameterValueException: The maximum number of iterations is not at least one.
    """

    if data.empty:
        raise EmptyDataFrameException("The input DataFrame is empty.")

    if any([x_train.size == 0, x_test.size == 0, y_train.size == 0, y_test.size == 0]):
        raise EmptyArrayException("All the input arrays must be non-empty.")

    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise InvalidArrayShapeException("Corresponding sample and label arrays must have the same length.")

    if max_iter <= 0:
        raise InvalidParameterValueException("The input value for maximum number of iterations must be at least one.")

    return _logistic_regression(data, x_train, x_test, y_train, y_test, penalty, max_iter)
