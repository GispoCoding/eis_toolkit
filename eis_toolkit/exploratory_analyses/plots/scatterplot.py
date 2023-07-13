from typing import Any, Literal, Optional, Sequence, Tuple, Union

import matplotlib
import pandas as pd
import seaborn as sns
from plot_utils import get_palette

from eis_toolkit import exceptions


def scatterplot(
    data: pd.DataFrame,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    size: Optional[str] = None,
    style: Optional[str] = None,
    palette: Union[str, Sequence[str], None] = None,
    legend: Literal["auto", "brief", "full", False] = "brief",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """
    Create a scatterplot.

    Acts as a wrapper for seaborn.scatterplot(), for further info see
    https://seaborn.pydata.org/generated/seaborn.scatterplot.162

    Args:
        data: Tidy ("long-form") dataframe where each column is a variable and each row is an observation.
        x: The name of the DataFrame column to be used on the x-axis.
        y: The name of the DataFrame column to be used on the y-axis.
        hue: Grouping variable that will produce elements with different colors. Can be either categorical or numeric,
            although color mapping will behave differently in latter case. Default is None.
        size: Grouping variable that will produce elements with different sizes. Can be either categorical or numeric,
            although size mapping will behave differently in latter case. Default is None.
        style: Grouping variable that will produce elements with different styles. Can have a numeric dtype but will
            always be treated as categorical. Default is None.
        palette: Method for choosing the colors to use when mapping the hue semantic.
            Default is "deep" with categorical color data and "viridis" with numerical color data.
        legend: How to draw the legend. If "brief", numeric hue and size variables will be represented with a sample
            of evenly spaced values. If "full", every group will get an entry in the legend. If "auto", choose between
            brief or full representation based on number of levels. If False, do not draw a legend. Default is "brief".
        title: Title for the figure. Default is None.
        xlabel: Label for the x-axis. Default is None, (column name of x-axis data).
        ylabel: Label for the y-axis. Default is None, (column name of y-axis data).
        xlim: The lower and upper range of the x-axis. Default is None.
        ylim: The lower and upper range of the y-axis. Default is None.
        **kwargs: Additional keyword arguments that are passed to seaborn.scatterplot() and the underlying
            matplotlib functions.

    Returns:
        A matplotlib axes containing the scatterplot.

    Raises:
        EmptyDataFrameException: Input DataFrame is empty.
        InvalidColumnException: A specified column name is not in the input DataFrame.
        InvalidParameterValueException: Either x or y is specified but not both.
    """
    if data.empty:
        raise exceptions.EmptyDataFrameException("Input DataFrame is empty.")
    for column_name in [x, y, hue, size, style]:
        if column_name is not None and column_name not in data.columns:
            raise exceptions.InvalidColumnException(f"'{column_name}' is not a column in the input DataFrame.")
    if (x is None and y is not None) or (x is not None and y is None):
        raise exceptions.InvalidParameterValueException("If x or y is specified, both need to be specified.")

    if palette:
        palette = get_palette(data, hue, palette)

    ax = sns.scatterplot(data=data, x=x, y=y, hue=hue, size=size, style=style, palette=palette, legend=legend, **kwargs)

    # Additional plot customization
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    return ax


# penguins = sns.load_dataset("penguins")
# scatterplot(penguins, x="bill_depth_mm", y="bill_length_mm", hue="island")
# plt.show()
