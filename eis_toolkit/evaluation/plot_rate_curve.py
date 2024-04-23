import matplotlib
import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Union
from matplotlib import pyplot as plt
from sklearn.metrics import auc

from eis_toolkit.exceptions import InvalidParameterValueException


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
    auc_value = str(round(auc(x_values, y_values), 2))
    plt.text(0.8, 0.2, "AUC: " + auc_value, bbox=auc_bbox)
    plt.title(label)
    fig.legend(bbox_to_anchor=(0.85, 0.4))

    return fig


@beartype
def plot_rate_curve(
    x_values: Union[np.ndarray, pd.Series],
    y_values: Union[np.ndarray, pd.Series],
    plot_title: str = "success_rate",
) -> matplotlib.figure.Figure:
    """Plot success rate.

    Y-axis is true positive rate and x-axis is proportion of area.

    Args:
        x_values: Proportion of area values.
        y_values: True positive rate values.
        plot_title: Success rate

    Returns:
        Matplotlib figure containing the produced plot.

    Raises:
        InvalidParameterValueException: Invalid plot type.
        InvalidParameterValueException: x_values or y_values are out of bounds.
    """
    label = plot_title
    xlab = "Proportion of area"

    if x_values.max() > 1 or x_values.min() < 0:
        raise InvalidParameterValueException("x_values should be within range 0-1")

    if y_values.max() > 1 or y_values.min() < 0:
        raise InvalidParameterValueException("y_values should be within range 0-1")

    fig = _plot_rate_curve(x_values=x_values, y_values=y_values, label=label, xlab=xlab)

    return fig
