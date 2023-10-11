import numpy as np
from sklearn.datasets import load_iris

from eis_toolkit.prediction.random_forests import (  # random_forest_regressor_predict,; random_forest_regressor_train,
    random_forest_classifier_predict,
    random_forest_classifier_train,
)

X, y = load_iris(return_X_y=True)


def test_random_forest_classifier():
    """Test that random forest classifier works as expected."""

    model, report_dict = random_forest_classifier_train(X, y, n_estimators=50, random_state=42)
    predicted_labels = random_forest_classifier_predict(model, X)

    np.testing.assert_equal(len(predicted_labels), len(y))

    # Test that all predicted labels have perfect metric scores since we are predicting with the test data
    labels = ["0", "1", "2"]
    metrics = ["precision", "recall", "f1-score"]
    for label in labels:
        for metric in metrics:
            np.testing.assert_equal(report_dict[label][metric], 1.0)
