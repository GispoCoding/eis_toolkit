from numbers import Number

import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Dict, Sequence, Union
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from eis_toolkit.exceptions import InvalidParameterValueException


@beartype
def score_predictions(
    y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series], metrics: Union[str, Sequence[str]]
) -> Union[Number, Dict[str, Number]]:
    """
    Score model predictions with given metrics.

    One or multiple metrics can be defined for scoring.

    Supported classifier metrics: "accuracy", "precision", "recall", "f1".
    Supported regressor metrics: "mse", "rmse", "mae", "r2".

    Args:
        y_true: Target values ("ground truth") against which scoring is performed.
        y_pred: Predicted labels.
        metrics: The metrics to use for scoring the model. Select only metrics applicable
            for the model type.

    Returns:
        Metric scores as a dictionary if multiple metrics, otherwise just the metric value.
    """
    if isinstance(metrics, str):
        score = _score_predictions(y_true, y_pred, metrics)
        return score
    else:
        out_metrics = {}
        for metric in metrics:
            score = _score_predictions(y_true, y_pred, metric)
            out_metrics[metric] = score
        return out_metrics


@beartype
def _score_predictions(
    y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series], metric: str
) -> Number:
    # Multiclass classification
    if len(y_true) > 2:
        average_method = "micro"
    # Binary classification
    else:
        average_method = "binary"

    if metric == "mae":
        score = mean_absolute_error(y_true, y_pred)
    elif metric == "mse":
        score = mean_squared_error(y_true, y_pred)
    elif metric == "rmse":
        score = mean_squared_error(y_true, y_pred, squared=False)
    elif metric == "r2":
        score = r2_score(y_true, y_pred)
    elif metric == "accuracy":
        score = accuracy_score(y_true, y_pred)
    elif metric == "precision":
        score = precision_score(y_true, y_pred, average=average_method)
    elif metric == "recall":
        score = recall_score(y_true, y_pred, average=average_method)
    elif metric == "f1":
        score = f1_score(y_true, y_pred, average=average_method)
    else:
        raise InvalidParameterValueException(f"Unrecognized metric: {metric}")

    return score
