from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from beartype.typing import Optional
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    DetCurveDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def summarize_probability_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Generate a comprehensive report of various evaluation metrics for classification probabilities.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities for the positive class. The array should come from
            a binary classifier.

    Returns:
        A dictionary containing the evaluated metrics.
    """
    metrics = {}

    metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    metrics["log_loss"] = log_loss(y_true, y_prob)
    metrics["average_precision"] = average_precision_score(y_true, y_prob)
    metrics["brier_score_loss"] = brier_score_loss(y_true, y_prob)

    return metrics


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    plot_title: Optional[str] = "ROC curve",
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot ROC (receiver operating characteristic) curve.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities for the positive class. The array should come from
            a binary classifier.
        plot_title: Title for the plot. Defaults to "ROC curve".
        ax: Existing Axes in which to draw the plot. Defaults to None.
        **kwargs: Additional keyword arguments passed to matplotlib.pyplot.plot.

    Returns:
        Matplotlib axes containing the plot.
    """
    display = RocCurveDisplay.from_predictions(y_true, y_prob, plot_chance_level=True, ax=ax, **kwargs)
    out_ax = display.ax_
    out_ax.set(xlabel="False positive rate", ylabel="True positive rate", title=plot_title)
    return out_ax


def plot_det_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    plot_title: Optional[str] = "DET curve",
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot DET (detection error tradeoff) curve.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities for the positive class. The array should come from
            a binary classifier.
        plot_title: Title for the plot. Defaults to "DET curve".
        ax: Existing Axes in which to draw the plot. Defaults to None.
        **kwargs: Additional keyword arguments passed to matplotlib.pyplot.plot.

    Returns:
        Matplotlib axes containing the plot.
    """
    display = DetCurveDisplay.from_predictions(y_true, y_prob, ax=ax, **kwargs)
    out_ax = display.ax_
    out_ax.set(xlabel="False positive rate", ylabel="True positive rate", title=plot_title)
    return out_ax


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    plot_title: Optional[str] = "Precision-Recall curve",
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot precision-recall curve.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities for the positive class. The array should come from
            a binary classifier.
        plot_title: Title for the plot. Defaults to "Precision-Recall curve".
        ax: Existing Axes in which to draw the plot. Defaults to None.
        **kwargs: Additional keyword arguments passed to matplotlib.pyplot.plot.

    Returns:
        Matplotlib axes containing the plot.
    """
    display = PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax, **kwargs)
    out_ax = display.ax_
    out_ax.set(xlabel="False positive rate", ylabel="True positive rate", title=plot_title)
    return out_ax


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 5,
    plot_title: Optional[str] = "Calibration curve",
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot calibration curve (aka realibity diagram).

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities for the positive class. The array should come from
            a binary classifier.
        plot_title: Title for the plot. Defaults to "Precision-Recall curve".
        ax: Existing Axes in which to draw the plot. Defaults to None.
        **kwargs: Additional keyword arguments passed to matplotlib.pyplot.plot.

    Returns:
        Matplotlib axes containing the plot.
    """
    display = CalibrationDisplay.from_predictions(y_true, y_prob, n_bins=n_bins, ax=ax, **kwargs)
    out_ax = display.ax_
    out_ax.set(xlabel="Mean predicted probability", ylabel="Fraction of positives", title=plot_title)
    return out_ax


def plot_predicted_probability_distribution(
    y_prob: np.ndarray,
    n_bins: int = 5,
    plot_title: Optional[str] = "Distribution of predicted probabilities",
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot a histogram of the predicted probabilities.

    Args:
        y_prob: Predicted probabilities for the positive class. The array should come from
            a binary classifier.

        ax: Existing Axes in which to draw the plot. Defaults to None.
        **kwargs: Additional keyword arguments passed to sns.histplot and matplotlib.

    Returns:
        Matplolib axes containing the plot.
    """
    sns.set_theme(style="white")
    plt.figure()
    out_ax = sns.histplot(y_prob, bins=n_bins, ax=ax, **kwargs)
    out_ax.set(xlabel="Predicted probability", ylabel="Count", title=plot_title)
    return out_ax