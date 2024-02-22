import matplotlib
import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Literal, Union
from matplotlib import pyplot as plt

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.validation.calculate_auc import calculate_auc


def _plot_rate_curve(
    x_values: Union[np.ndarray, pd.Series], y_values: Union[np.ndarray, pd.Series], label: str, xlab: str
) -> matplotlib.figure.Figure:
    fig = plt.figure(figsize=(10, 7))
    plt.plot(x_values, y_values, label=label)
    plt.xlim(-0.01, 1)
    plt.ylim(0, 1.01)
    plt.xlabel(xlab)
    plt.ylabel("True positive rate")
    plt.plot([0, 1], [0, 1], "--", label="Random baseline")
    auc_bbox = dict(boxstyle="round", facecolor="grey", alpha=0.2)
    auc = str(round(calculate_auc(x_values, y_values), 2))
    plt.text(0.8, 0.2, "AUC: " + auc, bbox=auc_bbox)
    plt.title(label)
    fig.legend(bbox_to_anchor=(0.85, 0.4))

    return fig


@beartype
def plot_rate_curve(
    x_values: Union[np.ndarray, pd.Series],
    y_values: Union[np.ndarray, pd.Series],
    plot_type: Literal["success_rate", "prediction_rate", "roc"] = "success_rate",
) -> matplotlib.figure.Figure:
    """Plot success rate, prediction rate or ROC curve.

    Plot type depends on plot_type argument. Y-axis is always true positive rate, while x-axis can be either false
    positive rate (roc) or proportion of area (success and prediction rate) depending on plot type.

    Args:
        x_values: False positive rate values or proportion of area values.
        y_values: True positive rate values.
        plot_type: Plot type. Can be either: "success_rate", "prediction_rate" or "roc".

    Returns:
        Success rate, prediction rate or ROC plot figure object.

    Raises:
        InvalidParameterValueException: Invalid plot type.
        InvalidParameterValueException: x_values or y_values are out of bounds.
    """
    if plot_type == "success_rate":
        label = "Success rate"
        xlab = "Proportion of area"
    elif plot_type == "prediction_rate":
        label = "Prediction rate"
        xlab = "Proportion of area"
    elif plot_type == "roc":
        label = "ROC"
        xlab = "False positive rate"
    else:
        raise InvalidParameterValueException("Invalid plot type")

    if x_values.max() > 1 or x_values.min() < 0:
        raise InvalidParameterValueException("x_values should be within range 0-1")

    if y_values.max() > 1 or y_values.min() < 0:
        raise InvalidParameterValueException("y_values should be within range 0-1")

    fig = _plot_rate_curve(x_values=x_values, y_values=y_values, label=label, xlab=xlab)

    return fig
