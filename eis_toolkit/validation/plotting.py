import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.validation.metrics import calculate_auc, calculate_base_metrics, get_pa_intersection


def plot_rate_curve(  # type: ignore[no-any-unimported]
    true_positive_rate_values: np.ndarray,
    proportion_of_area_values: np.ndarray,
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
        InvalidParameterValueException: Invalid plot type
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

    fig = plt.figure(figsize=(10, 7))
    plt.plot(proportion_of_area_values, true_positive_rate_values, label=label)
    plt.xlim(0, 1)
    plt.ylim(0, 1.01)
    plt.xlabel(xlab)
    plt.ylabel("True positive rate")
    plt.plot([0, 1], [0, 1], "--", label="Random baseline")
    props = dict(boxstyle="round", facecolor="grey", alpha=0.2)
    auc = str(round(calculate_auc(true_positive_rate_values, proportion_of_area_values), 2))
    plt.text(0.8, 0.2, "AUC: " + auc, bbox=props)
    plt.title(label)
    fig.legend(bbox_to_anchor=(0.85, 0.4))

    return fig


def plot_prediction_area_curve(  # type: ignore[no-any-unimported]
    true_positive_rate_values: np.ndarray, proportion_of_area_values: np.ndarray, threshold_values: np.ndarray
) -> matplotlib.figure.Figure:
    """Plot prediction-area (P-A) plot.

    Plots prediction area plot that can be used to evaluate mineral prospectivity maps and evidential layers. See e.g.,
    Yousefi and Carranza (2015).

    Args:
        true_positive_rate_values (np.ndarray): True positive rate values.
        proportion_of_area_values (np.ndarray): Proportion of area values.
        threshold_values (np.ndarray): Threshold values.

    Returns:
        matplotlib.figure.Figure: P-A plot.

    References:
        Yousefi, Mahyar, and Emmanuel John M. Carranza. "Fuzzification of continuous-value spatial evidence for mineral
        prospectivity mapping." Computers & Geosciences 74 (2015): 97-109.
    """

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


import geopandas
import pandas as pd
import rasterio

rast = rasterio.open("tests/data/remote/small_raster.tif")

df = pd.DataFrame(
    {
        "x": [384824, 384803, 384807, 384793, 384773, 384785],
        "y": [6671284, 6671295, 6671277, 6671293, 6671343, 6671357],
    }
)

gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.x, df.y, crs="EPSG:3067"))
coords = calculate_base_metrics(rast, gdf)
plot_rate_curve(coords.true_positive_rate, coords.proportion_of_area, "prediction_rate")
i = get_pa_intersection(coords.true_positive_rate, coords.proportion_of_area, coords.threshold)
plot_prediction_area_curve(coords.true_positive_rate, coords.proportion_of_area, coords.threshold)
