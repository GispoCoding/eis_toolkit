import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Tuple, Union
from sklearn.base import BaseEstimator, is_classifier
from tensorflow import keras

from eis_toolkit.exceptions import InvalidModelTypeException


@beartype
def predict_classifier(
    data: Union[np.ndarray, pd.DataFrame],
    model: Union[BaseEstimator, keras.Model],
    classification_threshold: float = 0.5,
    include_probabilities: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Predict with a trained classifier model.

    Only works for binary classification currently.

    Args:
        data: Data used to make predictions.
        model: Trained classifier or regressor. Can be any machine learning model trained with
            EIS Toolkit (Sklearn and Keras models).
        classification_threshold: Threshold for classifying based on probabilities. Defaults to 0.5.
        include_probabilities: If the probability array should be returned too. Defaults to True.

    Returns:
        Predicted labels and optionally predicted probabilities as one-dimensional arrays by a classifier model.

    Raises:
        InvalidModelTypeException: Input model is not a classifier model.
    """
    if isinstance(model, keras.Model):
        probabilities = model.predict(data).squeeze()
        labels = probabilities >= classification_threshold
        if include_probabilities:
            return labels, probabilities.astype(np.float32)
        else:
            return labels
    elif isinstance(model, BaseEstimator):
        if not is_classifier(model):
            raise InvalidModelTypeException(f"Expected a classifier model: {type(model)}.")
        probabilities = model.predict_proba(data)[:, 1]
        labels = (probabilities >= classification_threshold).astype(np.float32)
        if include_probabilities:
            return labels, probabilities.astype(np.float32)
        else:
            return labels
    else:
        raise InvalidModelTypeException(f"Model type not recognized: {type(model)}.")


@beartype
def predict_regressor(
    data: Union[np.ndarray, pd.DataFrame],
    model: Union[BaseEstimator, keras.Model],
) -> np.ndarray:
    """
    Predict with a trained regressor model.

    Args:
        data: Data used to make predictions.
        model: Trained classifier or regressor. Can be any machine learning model trained with
            EIS Toolkit (Sklearn and Keras models).

    Returns:
        Regression model prediction array.

    Raises:
        InvalidModelTypeException: Input model is not a regressor model.
    """
    if is_classifier(model):
        raise InvalidModelTypeException(f"Expected a regressor model: {type(model)}.")
    result = model.predict(data)
    return result
