from numbers import Number
from typing import Any, Literal, Optional, Sequence, Tuple, Union

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from plot_utils import get_palette

from eis_toolkit import exceptions


def histogram(
    data: Union[pd.DataFrame, np.ndarray, Sequence],
    x: Optional[str] = None,
    hue: Optional[str] = None,
    weights: Optional[str] = None,
    stat: str = "count",
    bins: Union[str, int] = "auto",
    binwidth: Optional[int] = None,
    binrange: Optional[Tuple[Number, Number]] = None,
    discrete: Optional[bool] = None,
    cumulative: bool = False,
    multiple: Literal["layer", "dodge", "stack", "fill"] = "layer",
    element: Literal["bars", "step", "poly"] = "bars",
    fill: bool = True,
    shrink: Number = 1,
    kde: bool = False,
    log_scale: Optional[bool] = None,
    palette: Union[str, Sequence[str], None] = None,
    legend: bool = True,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """
    Draw a univariate histogram with seaborn.

    Acts as a wrapper for seaborn.histplot(), for further info see https://seaborn.pydata.org/generated/seaborn.histplot

    Args:
        data: Input data structure. Should be either DataFrame, Numpy array or a sequence.
        x: The name of the DataFrame column to be used on the x-axis.
        hue: Grouping variable that will produce elements with different colors.
        weights: An optional array of weights, of the same shape as `x`.
        stat: Aggregate statistic to compute in each bin. `Count` shows the number of observations, `frequency` shows
            the number of observations divided by the bin width, `density` normalizes counts so that the area of the
            histogram is 1, `probability` normalizes counts so that the sum of the bar heights is 1.
        bins: Specification of hist bins, or None to use Freedman-Diaconis rule.
        binwidth: Size of each bin.
        binrange: Lowest and highest values of bins.
        discrete: If True, draw a bar at each unique value in `x` and extend the frequency of the first bin to 0.
        cumulative: If True, the histogram accumulates the values from left to right.
        multiple: Approach to resolving multiple elements when semantic mapping creates subsets.
        element: Element to draw for each bin.
        fill: If True, fill the bars in the histogram.
        shrink: Scale factor for shrinking the bins.
        kde: If True, compute a kernel density estimate to smooth the distribution and show on the plot.
        log_scale: If True, plot on log scale.
        palette: Method for choosing the colors to use when mapping the hue semantic.
        legend: If legend is included.
        title: Title for the plot.
        xlabel: Label for the x-axis. If None, the column name of x will be used.
        ylabel: Label for the y-axis. If None, depends on the `stat` parameter.
        **kwargs: Additional keyword arguments that are passed to seaborn.histplot() and the underlying
            matplotlib functions.

    Returns:
        The matplotlib axes containing the plot.

    Raises:
        EmptyDataFrameException: Input DataFrame is empty.
        InvalidColumnException: A specified column name is not in the input DataFrame.
        InvalidParameterValueException: DataFrame parameters specified when input data is not a DataFrame.
    """
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise exceptions.EmptyDataFrameException("Input DataFrame is empty.")
        for column_name in [x, hue, weights]:
            if column_name is not None and column_name not in data.columns:
                raise exceptions.InvalidColumnException(f"'{column_name}' is not a column in the input DataFrame.")
    elif x is not None or hue is not None or weights is not None:
        raise exceptions.InvalidParameterValueException(
            "When 'data' is not a DataFrame, 'x', 'hue', and 'weights' should not be specified."
        )

    if palette:
        palette = get_palette(data, hue, palette)

    ax = sns.histplot(
        data=data,
        x=x,
        hue=hue,
        weights=weights,
        stat=stat,
        bins=bins,
        binwidth=binwidth,
        binrange=binrange,
        discrete=discrete,
        cumulative=cumulative,
        multiple=multiple,
        element=element,
        fill=fill,
        shrink=shrink,
        kde=kde,
        log_scale=log_scale,
        palette=palette,
        legend=legend,
        **kwargs,
    )

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    return ax


def histogram_bivariate(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    weights: Optional[str] = None,
    stat: str = "count",
    bins: Tuple[Union[str, int], Union[str, int]] = ("auto", "auto"),
    binwidth: Optional[Tuple[int, int]] = None,
    binrange: Optional[Tuple[Tuple[Number, Number], Tuple[Number, Number]]] = None,
    discrete: Tuple[bool, bool] = (False, False),
    cumulative: bool = False,
    log_scale: Tuple[bool, bool] = (False, False),
    palette: Union[str, Sequence[str], None] = None,
    legend: Literal["auto", "brief", "full", False] = "brief",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """
    Draw a bivariate histogram with seaborn.

    Acts as a wrapper for seaborn.histplot(), for further info see https://seaborn.pydata.org/generated/seaborn.histplot

    For bins, binwidth, binrange, discrete and log_scale, a tuple is expected (values for x- and y-axis separately).

    Args:
        data: Input DataFrame.
        x: The name of the DataFrame column to be used on the x-axis.
        y: The name of the DataFrame column to be used on the y-axis.
        hue: Grouping variable that will produce elements with different colors.
        weights: An optional array of weights, of the same shape as `x`.
        stat: Aggregate statistic to compute in each bin. `Count` shows the number of observations, `frequency` shows
            the number of observations divided by the bin width, `density` normalizes counts so that the area of the
            histogram is 1, `probability` normalizes counts so that the sum of the bar heights is 1.
        bins: Specification of hist bins, or None to use Freedman-Diaconis rule.
        bins: Specification of hist bins, or None to use Freedman-Diaconis rule.
        binwidth: Size of each bin.
        binrange: Lowest and highest values of bins.
        discrete: If True, draw a bar at each unique value in `x` and extend the frequency of the first bin to 0.
        cumulative: If True, the histogram accumulates the values from left to right.
        multiple: Approach to resolving multiple elements when semantic mapping creates subsets.
        element: Element to draw for each bin.
        fill: If True, fill the bars in the histogram.
        shrink: Scale factor for shrinking the bins.
        kde: If True, compute a kernel density estimate to smooth the distribution and show on the plot.
        log_scale: If True, plot on log scale.
        palette: Method for choosing the colors to use when mapping the hue semantic.
        legend: If legend is included.
        title: Title for the plot.
        xlabel: Label for the x-axis. If None, the column name of x will be used.
        ylabel: Label for the y-axis. If None, depends on the `stat` parameter.
        **kwargs: Additional keyword arguments that are passed to seaborn.histplot() and the underlying
            matplotlib functions.

    Returns:
        The matplotlib axes containing the plot.

    Raises:
        EmptyDataFrameException: Input DataFrame is empty.
        InvalidColumnException: A specified column name is not in the input DataFrame.
    """

    if data.empty:
        raise exceptions.EmptyDataFrameException("Input DataFrame is empty.")
    for data_column in [x, y]:
        if data_column not in data.columns:
            raise exceptions.InvalidColumnException(f"'{data_column}' is not a column in the input DataFrame.")
    for column_name in [hue, weights]:
        if column_name is not None and column_name not in data.columns:
            raise exceptions.InvalidColumnException(f"'{column_name}' is not a column in the input DataFrame.")

    if palette:
        palette = get_palette(data, hue, palette)

    ax = sns.histplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        weights=weights,
        stat=stat,
        bins=bins,
        binwidth=binwidth,
        binrange=binrange,
        discrete=discrete,
        cumulative=cumulative,
        log_scale=log_scale,
        palette=palette,
        legend=legend,
        **kwargs,
    )

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    return ax
