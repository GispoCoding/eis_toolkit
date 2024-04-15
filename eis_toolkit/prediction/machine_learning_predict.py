import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Tuple, Union
from sklearn.base import BaseEstimator
from tensorflow import keras


@beartype
def predict_classifier(
    data: Union[np.ndarray, pd.DataFrame], model: Union[BaseEstimator, keras.Model], include_probabilities: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Predict with a trained model.

    Args:
        data: Data used to make predictions.
        model: Trained classifier or regressor. Can be any machine learning model trained with
            EIS Toolkit (Sklearn and Keras models).
        include_probabilities: If the probability array should be returned too. Defaults to True.

    Returns:
        Predicted labels and optionally predicted probabilities by a classifier model.
    """
    if isinstance(model, keras.Model):
        probabilities = model.predict(data)
        labels = probabilities.argmax(axis=-1)
        if include_probabilities:
            return labels, probabilities
        else:
            return labels
    elif include_probabilities:
        probabilities = model.predict_proba(data)
        labels = model.predict(data)
        return labels, probabilities
    else:
        labels = model.predict(data)
        return labels


@beartype
def predict_regressor(
    data: Union[np.ndarray, pd.DataFrame],
    model: Union[BaseEstimator, keras.Model],
) -> np.ndarray:
    """
    Predict with a trained model.

    Args:
        data: Data used to make predictions.
        model: Trained classifier or regressor. Can be any machine learning model trained with
            EIS Toolkit (Sklearn and Keras models).

    Returns:
        Regression model prediction array.
    """
    result = model.predict(data)
    return result
