import numpy as np
import pandas as pd
import sklearn.neural_network
from beartype import beartype
from beartype.typing import Optional, Sequence
from sklearn.inspection import permutation_importance

from eis_toolkit.exceptions import (
    InvalidDatasetException,
    InvalidParameterValueException,
    NonMatchingParameterLengthsException,
)


@beartype
def evaluate_feature_importance(
    model: sklearn.base.BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    n_repeats: int = 10,
    random_state: Optional[int] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate the feature importance of a Sklearn classifier or regressor.

    Args:
        model: A trained and fitted Sklearn model.
        X: Feature data.
        y: Target labels.
        feature_names: Names of features in X.
        n_repeats: Number of iteration used when calculating feature importance. Defaults to 10.
        random_state:  Seed for random number generation. Defaults to None.

    Returns:
        A dataframe containing features and their importance.
        A dictionary containing importance mean, importance std, and overall importance.

    Raises:
        InvalidDatasetException: Either array is empty.
        InvalidParameterValueException: Value for 'n_repeats' is not at least one.
    """

    if X.size == 0:
        raise InvalidDatasetException("Feature matrix X is empty.")

    if y.size == 0:
        raise InvalidDatasetException("Target labels y is empty.")

    if n_repeats < 1:
        raise InvalidParameterValueException("Value for 'n_repeats' is less than one.")

    if len(X) != len(y):
        raise NonMatchingParameterLengthsException("Feature matrix X and target labels y must have the same length.")

    if len(feature_names) != X.shape[1]:
        raise InvalidParameterValueException("Number of feature names must match the number of input features.")

    result = permutation_importance(model, X, y.ravel(), n_repeats=n_repeats, random_state=random_state)

    feature_importance = pd.DataFrame({"Feature": feature_names, "Importance": result.importances_mean})

    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

    return feature_importance, result
