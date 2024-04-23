import matplotlib
import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import Tuple, Union
from matplotlib import pyplot as plt
from shapely.geometry import LineString

from eis_toolkit.exceptions import InvalidParameterValueException


def _get_pa_intersection(
    true_positive_rate_values: Union[np.ndarray, pd.Series],
    proportion_of_area_values: Union[np.ndarray, pd.Series],
    threshold_values: Union[np.ndarray, pd.Series],
) -> Tuple[float, float]:
    true_positive_area_curve = LineString(np.column_stack((threshold_values, true_positive_rate_values)))
    proportion_of_area_values_curve = LineString(np.column_stack((threshold_values, 1 - proportion_of_area_values)))
    intersection = true_positive_area_curve.intersection(proportion_of_area_values_curve)

    return intersection.x, intersection.y


def _plot_prediction_area_curves(
    true_positive_rate_values: Union[np.ndarray, pd.Series],
    proportion_of_area_values: Union[np.ndarray, pd.Series],
    threshold_values: Union[np.ndarray, pd.Series],
) -> matplotlib.figure.Figure:
    x, y = _get_pa_intersection(true_positive_rate_values, proportion_of_area_values, threshold_values)

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(threshold_values, true_positive_rate_values, "r-", label="Prediction rate")

    ax2.plot(threshold_values, proportion_of_area_values, "b-", label="Area")
    ax2.plot(x, 1 - y, " o", markersize=7, c="black", label="Intersection point")
    ax1.set_ylim(0, 1.01)
    ax2.set_ylim(-0.01, 1)
    ax2.invert_yaxis()
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("True positive rate", color="r")
    ax2.set_ylabel("Proportion of area", color="b")
    ax1.annotate(
        text="TPR:" + str(round(y, 2)),
        xy=(x, y),
        xytext=(x + threshold_values.max() / 10, y),
        arrowprops=dict(facecolor="black", shrink=0.09, width=0.3),
        verticalalignment="center",
    )
    fig.legend(bbox_to_anchor=(0.3, 0.6), bbox_transform=ax1.transAxes)
    plt.title("Prediction-area plot")

    return fig


@beartype
def plot_prediction_area_curves(
    true_positive_rate_values: Union[np.ndarray, pd.Series],
    proportion_of_area_values: Union[np.ndarray, pd.Series],
    threshold_values: Union[np.ndarray, pd.Series],
) -> matplotlib.figure.Figure:
    """Plot prediction-area (P-A) plot.

    Plots prediction area plot that can be used to evaluate mineral prospectivity maps and evidential layers. See e.g.,
    Yousefi and Carranza (2015).

    The inputs needed for this tool can be obtained with calculate_base_metrics() tool.

    Args:
        true_positive_rate_values: True positive rate values.
        proportion_of_area_values: Proportion of area values.
        threshold_values: Threshold values.

    Returns:
        P-A plot figure object.

    Raises:
        InvalidParameterValueException: true_positive_rate_values or proportion_of_area_values values are out of bounds.

    References:
        Yousefi, Mahyar, and Emmanuel John M. Carranza. "Fuzzification of continuous-value spatial evidence for mineral
        prospectivity mapping." Computers & Geosciences 74 (2015): 97-109.
    """
    if true_positive_rate_values.max() > 1 or true_positive_rate_values.min() < 0:
        raise InvalidParameterValueException("true_positive_rate values should be within range 0-1")

    if proportion_of_area_values.max() > 1 or proportion_of_area_values.min() < 0:
        raise InvalidParameterValueException("proportion_of_area values should be within range 0-1")

    fig = _plot_prediction_area_curves(
        true_positive_rate_values=true_positive_rate_values,
        proportion_of_area_values=proportion_of_area_values,
        threshold_values=threshold_values,
    )
    return fig
