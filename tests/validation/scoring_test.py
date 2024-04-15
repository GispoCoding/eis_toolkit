from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from eis_toolkit.prediction.machine_learning_predict import predict_classifier
from eis_toolkit.prediction.random_forests import random_forest_classifier_train
from eis_toolkit.validation.scoring import score_predictions

X, y = make_classification(n_samples=200, n_features=20, n_informative=2, n_redundant=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
rf_model, history = random_forest_classifier_train(X_train, y_train)
y_pred = predict_classifier(X_test, rf_model, include_probabilities=False)


def test_scoring_one_metric():
    """Tests that scoring predictions with one metric works as expected."""
    score = score_predictions(y_test, y_pred, "accuracy")
    assert isinstance(score, float)


def test_scoring_multiple_metrics():
    """Tests that scoring predictions with multuple metrics works as expected."""
    scores = score_predictions(y_test, y_pred, ["accuracy", "precision", "recall"])
    assert isinstance(scores, dict)
    assert len(scores) == 3
