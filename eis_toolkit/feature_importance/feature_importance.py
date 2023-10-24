import numpy as np
import pandas
import pandas as pd
import sklearn.neural_network
from sklearn.inspection import permutation_importance

from eis_toolkit.exceptions import InvalidDatasetException


def evaluate_feature_importance(
    clf: sklearn.neural_network or sklearn.linear_model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    number_of_repetition: int = 50,
    random_state: int = 0,
) -> (pandas.DataFrame, dict):
    """
    Evaluate the feature importance of a sklearn classifier or linear model.

    Parameters:
        clf (Any sklearn nn model or lm model): Trained classifier.
        x_test (np.ndarray): Testing feature data (X data need to be normalized / standardized).
        y_test (np.ndarray): Testing target data.
        feature_names (list): Names of the feature columns.
        number_of_repetition (int): Number of iteration used when calculate feature importance (default 50).
        random_state (int): random state for repeatability of results (Default 0).
    Return:
        feature_importance (pd.Dataframe): A dataframe composed by features name and Importance value
        result (dict[object]): The resulted object with importance mean, importance std, and overall importance
    Raise:
        InvalidDatasetException: When the dataset is None.
    """

    if x_test is None or y_test is None:
        raise InvalidDatasetException

    result = permutation_importance(
        clf, x_test, y_test.ravel(), n_repeats=number_of_repetition, random_state=random_state
    )

    feature_importance = pd.DataFrame({"Feature": feature_names, "Importance": result.importances_mean})

    feature_importance["Importance"] = feature_importance["Importance"] * 100
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)
    # feature_importance['Importance'] = feature_importance['Importance'].apply(lambda x: '{:.6f}%'.format(x))

    return feature_importance, result
