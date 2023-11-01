from typing import Sequence

import numpy as np
import pandas as pd
import sklearn.neural_network
from beartype import beartype
from sklearn.inspection import permutation_importance

from eis_toolkit.exceptions import InvalidDatasetException


@beartype
def evaluate_feature_importance(
    classifier: sklearn.base.BaseEstimator,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Sequence[str],
    number_of_repetition: int = 50,
    random_state: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate the feature importance of a sklearn classifier or linear model.

    Parameters:
        classifier: Trained classifier.
        x_test: Testing feature data (X data need to be normalized / standardized).
        y_test: Testing target data.
        feature_names: Names of the feature columns.
        number_of_repetition: Number of iteration used when calculate feature importance (default 50).
        random_state: random state for repeatability of results (Default 0).
    Return:
        A dataframe composed by features name and Importance value
        The resulted object with importance mean, importance std, and overall importance
    Raises:
        InvalidDatasetException: When the dataset is None.
    """

    if x_test is None or y_test is None:
        raise InvalidDatasetException

    result = permutation_importance(
        classifier, x_test, y_test.ravel(), n_repeats=number_of_repetition, random_state=random_state
    )

    feature_importance = pd.DataFrame({"Feature": feature_names, "Importance": result.importances_mean})

    feature_importance["Importance"] = feature_importance["Importance"] * 100
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

    return feature_importance, result
