import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.validation.calculate_auc import calculate_auc


def _plot_rate_curve(  # type: ignore[no-any-unimported]
    x_values: np.ndarray, y_values: np.ndarray, label: str, xlab: str
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


def plot_rate_curve(  # type: ignore[no-any-unimported]
    x_values: np.ndarray,
    y_values: np.ndarray,
    plot_type: str = "success_rate",
) -> matplotlib.figure.Figure:
    """Plot success or prediction rate curve.

    Type of plot depends on the given deposits. If deposits were used for model training, then the plot is known as
    success rate curve. If deposits were not used for model training then the plot is known as prediction rate plot. In
    both cases x-axis indicates the proportion of area that is considired to be prospective and y-axis indicates true
    positive rate.

    Args:
        true_positive_rate_values (np.ndarray): True positive rate values, y-coordinates of the plot.
        proportion_of_area_values (np.ndarray): Proportion of area values, x-coordinates of the plot.
        plot_type (str): Plot type. Can be either: "success_rate", "prediction_rate" or "roc".

    Returns:
        matplotlib.figure.Figure: Success rate, prediction rate or ROC plot.

    Raises:
        InvalidParameterValueException: Invalid plot type.
        ValueError: x_values or y_values are out of bounds.
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
        raise ValueError("x_values should be within range 0-1")

    if y_values.max() > 1 or y_values.min() < 0:
        raise ValueError("y_values should be within range 0-1")

    fig = _plot_rate_curve(x_values=x_values, y_values=y_values, label=label, xlab=xlab)

    return fig
