import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

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
    "Em_Qd",
    "EM_Inph",
]

X = pd.read_csv("./tests/data/remote/fake_smote_data.csv").to_numpy()
X = StandardScaler().fit_transform(X)
labels = np.random.randint(2, size=13)


def test_check_label_and_data_dimension():
    """Let s check if labels and X data agrees."""
    assert labels.shape[0] == X.shape[0]


def test_instance_model():
    """Here I check if I can have an instance of the model."""
    classifier = MLPClassifier(solver="adam", alpha=0.001, hidden_layer_sizes=(16, 8), random_state=0, max_iter=500)
    classifier.fit(X, labels.ravel())
    feature_importance, dict_of_results = evaluate_feature_importance(
        classifier=classifier, x_test=X, y_test=labels, feature_names=feature_names
    )


def test_model_results():
    """Test the model and check if there is results as output."""
    classifier = MLPClassifier(solver="adam", alpha=0.001, hidden_layer_sizes=(16, 8), random_state=0, max_iter=500)
    classifier.fit(X, labels.ravel())
    feature_importance, dict_of_results = evaluate_feature_importance(
        classifier=classifier, x_test=X, y_test=labels, feature_names=feature_names
    )

    assert len(feature_importance) > 0
    assert len(dict_of_results.keys()) > 0
