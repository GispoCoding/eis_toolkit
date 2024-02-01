import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from beartype.typing import Optional

from eis_toolkit.exceptions import EmptyDataFrameException


def plot_correlation_matrix(
    matrix: pd.DataFrame,
    annotate: bool = True,
    cmap: Optional[matplotlib.colors.ListedColormap] = None,
    plot_title: Optional[str] = None,
    **kwargs: dict
) -> matplotlib.axes.Axes:
    """
    Create a Seaborn heatmap to visualize correlation matrix.

    Args:
        matrix: Correlation matrix as a DataFrame.
        annotate: If plot squares should display the correlation values. Defaults to True.
        cmap: Colormap for plotting. Optional parameter. Defaults to None, in which
            case a default colormap is used.
        plot_title: Title of the plot. Optional parameter, defaults to none (no title).
        **kwargs: Additional parameters to pass to Seaborn and matplotlib.

    Returns:
        Matplotlib axes object with the produced plot.

    Raises:
        EmptyDataFrameException: Input matrix is empty.
    """
    if matrix.empty:
        raise EmptyDataFrameException("Input matrix DataFrame is empty.")

    # Mask for the upper triangle of the heatmap
    mask = np.triu(np.ones_like(matrix, dtype=bool))

    if cmap is None:
        # Generate a default diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

    ax = sns.heatmap(
        matrix,
        mask=mask,
        cmap=cmap,
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        annot=annotate,
        cbar_kws={"shrink": 0.5},
        **kwargs
    )
    if plot_title is not None:
        ax.set_title(plot_title)

    return ax
