import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from eis_toolkit.exceptions import InvalidArgumentTypeException
from eis_toolkit.prediction.mlp import train_evaluate_predict_with_mlp


def test_the_invalid_argument_exception():
    """This check test if the exception is throws correctly."""
    X = pd.read_csv("../data/remote/fake_smote_data.csv").to_numpy()
    X = StandardScaler().fit_transform(X)
    labels = np.random.randint(2, size=X.shape[0])
    with pytest.raises(InvalidArgumentTypeException):
        train_evaluate_predict_with_mlp(
            dataset=X,
            labels=labels,
            cross_validation_type="SKFOLD",
            number_of_split=5,
            is_class_probability=True,
            is_predict_full_map=False,
        )


def test_check_prediction_is_not_empty():
    """Check if the final prediction are not empty."""
    X = pd.read_csv("../data/remote/fake_smote_data.csv").to_numpy()
    X = StandardScaler().fit_transform(X)
    labels = np.random.randint(2, size=X.shape[0])
    prediction = train_evaluate_predict_with_mlp(
        dataset=X,
        labels=labels,
        cross_validation_type="SKFOLD",
        number_of_split=5,
        is_class_probability=False,
        is_predict_full_map=False,
    )
    assert len(prediction) > 0
