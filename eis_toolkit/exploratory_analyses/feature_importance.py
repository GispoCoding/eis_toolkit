import numpy as np
import pandas as pd
import sklearn.neural_network
from beartype import beartype
from beartype.typing import Optional, Sequence
from sklearn.inspection import permutation_importance

from eis_toolkit.exceptions import InvalidDatasetException, InvalidParameterValueException


@beartype
def evaluate_feature_importance(
    model: sklearn.base.BaseEstimator,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Sequence[str],
    n_repeats: int = 50,
    random_state: Optional[int] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate the feature importance of a sklearn classifier or regressor.

    Args:
        model: A trained and fitted Sklearn model.
        x_test: Testing feature data (X data need to be normalized / standardized).
        y_test: Testing label data.
        feature_names: Names of the feature columns.
        n_repeats: Number of iteration used when calculate feature importance. Defaults to 50.
        random_state: random state for repeatability of results. Optional parameter.

    Returns:
        A dataframe containing features and their importance.
        A dictionary containing importance mean, importance std, and overall importance.

    Raises:
        InvalidDatasetException: Either array is empty.
        InvalidParameterValueException: Value for 'n_repeats' is not at least one.
    """

    if x_test.size == 0:
        raise InvalidDatasetException("Array 'x_test' is empty.")

    if y_test.size == 0:
        raise InvalidDatasetException("Array 'y_test' is empty.")

    if n_repeats < 1:
        raise InvalidParameterValueException("Value for 'n_repeats' is less than one.")

    result = permutation_importance(model, x_test, y_test.ravel(), n_repeats=n_repeats, random_state=random_state)

    feature_importance = pd.DataFrame({"Feature": feature_names, "Importance": result.importances_mean * 100})

    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

    return feature_importance, result
