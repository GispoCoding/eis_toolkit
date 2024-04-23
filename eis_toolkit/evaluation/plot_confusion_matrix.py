import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from beartype import beartype
from beartype.typing import Optional, Sequence, Union
from matplotlib.colors import Colormap

from eis_toolkit.exceptions import InvalidDataShapeException


@beartype
def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    cmap: Optional[Union[str, Colormap, Sequence]] = None,
    plot_title: str = "Confusion matrix",
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """Plot confusion matrix to visualize classification results.

    Args:
        confusion_matrix: The confusion matrix as 2D Numpy array. Expects the first element
            (upper-left corner) to have True negatives.
        cmap: Colormap name, matploltib colormap objects or list of colors for coloring the plot.
            Optional parameter.
        plot_title: Title for the plot. Defaults to "Confusion matrix".
        ax: An existing Axes in which to draw the plot. Defaults to None.
        **kwargs: Additional keyword arguments passed to sns.heatmap.

    Returns:
        Matplotlib axes containing the plot.

    Raises:
        InvalidDataShapeException: Raised if input confusion matrix is not square.
    """
    shape = confusion_matrix.shape
    if shape[0] != shape[1]:
        raise InvalidDataShapeException(f"Expected confusion matrix to be square, input array has shape: {shape}")
    names = None

    counts = ["{0:0.0f}".format(value) for value in confusion_matrix.flatten()]
    percentages = ["{0:.2%}".format(value) for value in confusion_matrix.flatten() / np.sum(confusion_matrix)]

    if shape == (2, 2):  # Binary classificaiton
        names = ["True Neg", "False Pos", "False Neg", "True Pos"]
        labels = np.asarray([f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(names, counts, percentages)]).reshape(shape)
    else:
        labels = np.asarray([f"{v1}\n{v2}" for v1, v2 in zip(counts, percentages)]).reshape(shape)

    out_ax = sns.heatmap(confusion_matrix, annot=labels, fmt="", cmap=cmap, ax=ax, **kwargs)
    out_ax.set(xlabel="Predicted label", ylabel="True label", title=plot_title)

    return out_ax
