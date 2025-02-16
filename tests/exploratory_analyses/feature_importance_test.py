import numpy as np
import pandas as pd
import pytest
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from eis_toolkit.exceptions import (
    InvalidDatasetException,
    InvalidParameterValueException,
    NonMatchingParameterLengthsException,
)
from eis_toolkit.exploratory_analyses.feature_importance import evaluate_feature_importance

feature_names = [
    "Mag_TMI",
    "Mag_AS",
    "DRC135",
    "DRC180",
    "DRC45",
    "DRC90",
    "Mag_TD",
    "HDTDR",
    "Mag_Xdrv",
    "mag_Ydrv",
    "Mag_Zdrv",
    "Pseu_Grv",
    "Rd_U",
    "Rd_TC",
    "Rd_Th",
    "Rd_K",
    "EM_ratio",
    "EM_Ap_rs",
    "EM_Qd",
    "EM_Inph",
]

data = pd.read_csv("./tests/data/remote/fake_smote_data.csv").to_numpy()
data = StandardScaler().fit_transform(data)
np.random.seed(0)
labels = np.random.randint(2, size=13)
classifier = MLPClassifier(solver="adam", alpha=0.001, hidden_layer_sizes=(16, 8), random_state=0, max_iter=500)


def test_empty_data():
    """Test that empty data or labels raise exception."""
    empty_data = np.array([])
    empty_labels = np.array([])
    with pytest.raises(InvalidDatasetException):
        _, _ = evaluate_feature_importance(model=classifier, X=empty_data, y=labels, feature_names=feature_names)

    with pytest.raises(InvalidDatasetException):
        _, _ = evaluate_feature_importance(model=classifier, X=data, y=empty_labels, feature_names=feature_names)


def test_invalid_n_repeats():
    """Test that invalid value for 'n_repeats' raises exception."""
    with pytest.raises(InvalidParameterValueException):
        _, _ = evaluate_feature_importance(model=classifier, X=data, y=labels, feature_names=feature_names, n_repeats=0)


def test_model_output():
    """Test that function output is as expected."""
    classifier.fit(data, labels.ravel())
    feature_importance, importance_results = evaluate_feature_importance(
        model=classifier, X=data, y=labels, feature_names=feature_names, n_repeats=50, random_state=0
    )

    np.testing.assert_almost_equal(
        feature_importance.loc[feature_importance["Feature"] == "EM_ratio", "Importance"].values[0],
        desired=0.129231,
        decimal=6,
    )
    np.testing.assert_almost_equal(
        feature_importance.loc[feature_importance["Feature"] == "EM_Qd", "Importance"].values[0],
        desired=0.044615,
        decimal=6,
    )
    np.testing.assert_equal(len(feature_importance), desired=len(feature_names))
    np.testing.assert_equal(
        tuple(importance_results.keys()),
        desired=("importances_mean", "importances_std", "importances"),
    )


def test_invalid_input_lengths():
    """Test that non matcing X and y lengths raises an exception."""
    labels = np.random.randint(2, size=12)
    with pytest.raises(NonMatchingParameterLengthsException):
        _, _ = evaluate_feature_importance(model=classifier, X=data, y=labels, feature_names=feature_names)


def test_invalid_number_of_feature_names():
    """Test that invalid number of feature names raises an exception."""
    with pytest.raises(InvalidParameterValueException):
        _, _ = evaluate_feature_importance(
            model=classifier,
            X=data,
            y=labels,
            feature_names=["a", "b", "c"],
        )
