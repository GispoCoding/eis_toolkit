from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from beartype import beartype
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from eis_toolkit import exceptions


@beartype
def logistic_regression_train(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    test_size: float = 0.25,
    penalty: Literal["l1", "l2", "elasicnet", None] = "l2",
    max_iter: int = 100,
    random_state: Optional[int] = None,
    solver: Literal["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"] = "lbfgs",
) -> Tuple[LogisticRegression, dict]:
    """
    Train a Logistic Regression classifier model using Sklearn.

    Trains the model with the given parameters and evaluates model performance using test data.

    The choice of the algorithm depends on the penalty chosen. Supported penalties by solver:
    'lbfgs' - ['l2', None]
    'liblinear' - ['l1', 'l2']
    'newton-cg' - ['l2', None]
    'newton-cholesky' - ['l2', None]
    'sag' - ['l2', None]
    'saga' - ['elasticnet', 'l1', 'l2', None]

    Args:
        X: Training data.
        y: Target labels.
        test_size: Fraction of the dataset to be used as test data (rest is used for training). Defaults to 0.25.
        penalty: Specifies the norm of the penalty. Defaults to 'l2'.
        max_iter: Maximum number of iterations taken for the solvers to converge. Defaults to 100.
        random_state: Seed for random number generation. Defaults to None.
        solver: Algorithm to use in the optimization problem. Defaults to 'lbfgs'.

    Returns:
        The trained Logistric Regression classifier and details of test set performance.

    Raises:
        NonMatchingParameterLengthsException: If length of X and y don't match.
        InvalidParameterValueException: test_size is not between 0 and 1 or max_iter is less than one.
    """
    x_size = X.index.size if isinstance(X, pd.DataFrame) else X.shape[0]
    if x_size != y.size:
        raise exceptions.NonMatchingParameterLengthsException(f"X and y must have the length {x_size} != {y.size}.")

    if not (0 <= test_size <= 1):
        raise exceptions.InvalidParameterValueException("Input value for test_size must be between 0 and 1.")

    if max_iter < 1:
        raise exceptions.InvalidParameterValueException("Input value for max_iter must be > 0.")

    # Splitting data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = LogisticRegression(penalty=penalty, max_iter=max_iter, random_state=random_state, solver=solver)

    model.fit(X_train, y_train)

    # Predictions for test data
    y_pred = model.predict(X_test)

    # Performance metrics
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, report


@beartype
def logistic_regression_predict(model: LogisticRegression, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Use a trained Logistic Regression model to make predictions.

    Args:
        model: Trained Logistic Regression classifier.
        X: Features for which predictions are to be made.

    Returns:
        Predicted labels.
    """
    predictions = model.predict(X)

    return predictions
