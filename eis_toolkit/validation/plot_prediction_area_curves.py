import matplotlib
import numpy as np
from beartype import beartype
from matplotlib import pyplot as plt

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.validation.get_pa_intersection import get_pa_intersection


def _plot_prediction_area_curves(  # type: ignore[no-any-unimported]
    true_positive_rate_values: np.ndarray, proportion_of_area_values: np.ndarray, threshold_values: np.ndarray
) -> matplotlib.figure.Figure:
    intersection = get_pa_intersection(true_positive_rate_values, proportion_of_area_values, threshold_values)

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(threshold_values, true_positive_rate_values, "r-", label="Prediction rate")

    ax2.plot(threshold_values, proportion_of_area_values, "b-", label="Area")
    ax2.plot(intersection[0], 1 - intersection[1], " o", markersize=7, c="black", label="Intersection point")
    ax1.set_ylim(0, 1.01)
    ax2.set_ylim(-0.01, 1)
    ax2.invert_yaxis()
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("True positive rate", color="r")
    ax2.set_ylabel("Proportion of area", color="b")
    ax1.annotate(
        text="TPR:" + str(round(intersection[1], 2)),
        xy=(intersection[0], intersection[1]),
        xytext=(intersection[0] + threshold_values.max() / 10, intersection[1]),
        arrowprops=dict(facecolor="black", shrink=0.09, width=0.3),
        verticalalignment="center",
    )
    fig.legend(bbox_to_anchor=(0.3, 0.6), bbox_transform=ax1.transAxes)
    plt.title("Prediction-area plot")

    return fig


@beartype
def plot_prediction_area_curves(  # type: ignore[no-any-unimported]
    true_positive_rate_values: np.ndarray, proportion_of_area_values: np.ndarray, threshold_values: np.ndarray
) -> matplotlib.figure.Figure:
    """Plot prediction-area (P-A) plot.

    Plots prediction area plot that can be used to evaluate mineral prospectivity maps and evidential layers. See e.g.,
    Yousefi and Carranza (2015).

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
