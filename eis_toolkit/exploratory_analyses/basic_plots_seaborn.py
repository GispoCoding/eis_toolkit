# from typing import Any, Literal, Optional, Sequence, Tuple, Union

# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns

# from eis_toolkit import exceptions


# def _get_palette(data, hue, palette):
#     # Check if hue column data is numeric
#     if np.issubdtype(data[hue].dtype, np.number):
#         # if palette is not specified by user, set it to 'viridis'
#         return palette if palette else "viridis"
#     else:
#         # if palette is not specified by user, set it to 'deep'
#         return palette if palette else "deep"


# def relplot(
#     data: pd.DataFrame,
#     x: Optional[str] = None,
#     y: Optional[str] = None,
#     hue: Optional[str] = None,
#     size: Optional[str] = None,
#     style: Optional[str] = None,
#     palette: Union[str, Sequence[str], None] = None,
#     col: Optional[str] = None,
#     row: Optional[str] = None,
#     col_wrap: Optional[int] = None,
#     legend: Literal["auto", "brief", "full", False] = "brief",
#     figsize: Tuple[int, int] = (10, 8),
#     title: Optional[str] = None,
#     xlabel: Optional[str] = None,
#     ylabel: Optional[str] = None,
#     xlim: Optional[Tuple[float, float]] = None,
#     ylim: Optional[Tuple[float, float]] = None,
#     **kwargs: Any,
# ) -> sns.FacetGrid:
#     """
#     Create a lineplot with the option of additional semantic groupings.

#     The relationship between `x` and `y` can be shown for different subsets of the data using the `hue`, `size`,
#     and `style` parameters. These parameters control what visual semantics are used to identify the different subsets.
#     It is possible to show up to three dimensions independently by using all three semantic types, but this style of
#     plot can be hard to interpret and is often ineffective. Using redundant semantics (i.e. both `hue` and `style` for
#     the same variable) can be helpful for making graphics more accessible.

#     Args:
#         data: Tidy ("long-form") dataframe where each column is a variable and each row is an observation.
#         x: The name of the DataFrame column to be used on the x-axis.
#         y: The name of the DataFrame column to be used on the y-axis.
#         hue: Grouping variable that will produce elements with different colors. Can be either categorical or numeric,
#             although color mapping will behave differently in latter case. Default is None.
#         size: Grouping variable that will produce elements with different sizes. Can be either categorical or numeric,
#             although size mapping will behave differently in latter case. Default is None.
#         style: Grouping variable that will produce elements with different styles. Can have a numeric dtype but will always be
#             treated as categorical. Default is None.
#         palette: Method for choosing the colors to use when mapping the hue semantic.
#             Default is "deep" with categorical color data and "viridis" with numerical color data.
#         col: Column name that will be used to produce multiple columns of plots. Default is None.
#         row: Column name that will be used to produce multiple rows of plots. Default is None.
#         col_wrap: The maximum number of facet columns for the grid. When more than `col_wrap` columns would be created,
#             a new row is started instead. Default is None.
#         legend: How to draw the legend. If "brief", numeric hue and size variables will be represented with a sample of evenly
#             spaced values. If "full", every group will get an entry in the legend. If "auto", choose between brief or full
#             representation based on number of levels. If False, do not draw a legend. Default is "brief".
#         figsize: Width and height of the figure. Default is (10, 8).
#         title: Title for the figure. Default is None.
#         xlabel: Label for the x-axis. Default is None, (column name of x-axis data).
#         ylabel: Label for the y-axis. Default is None, (column name of y-axis data).
#         xlim: The lower and upper range of the x-axis. Default is None.
#         ylim: The lower and upper range of the y-axis. Default is None.
#         **kwargs: Additional keyword arguments that are passed to seaborn.relplot() and the underlying
#             matplotlib functions.

#     Returns:
#         A seaborn.FacetGrid object with the lineplot(s) drawn onto it.

#     Raises:
#         EmptyDataFrameException: Input DataFrame is empty.
#         InvalidColumnException: A specified column name is not in the input DataFrame.
#     """
#     if data.empty:
#         raise exceptions.EmptyDataFrameException("Input DataFrame is empty.")
#     for column_name in [x, y, hue, size, style, col, row]:
#         if column_name is not None and column_name not in data.columns:
#             raise exceptions.InvalidColumnException(f"'{column_name}' is not a column in the input DataFrame.")
#     if (x is None and y is not None) or (x is not None and y is None):
#         raise exceptions.InvalidParameterValueException("If x or y is specified, both need to be specified.")

#     if palette:
#         palette = _get_palette(data, hue, palette)

#     # Force line type
#     kwargs["kind"] = "line"
#     facet_grid = sns.relplot(
#         data=data,
#         x=x,
#         y=y,
#         hue=hue,
#         size=size,
#         style=style,
#         palette=palette,
#         col=col,
#         row=row,
#         col_wrap=col_wrap,
#         legend=legend,
#         **kwargs,
#     )

#     # Additional plot customization
#     facet_grid.fig.set_size_inches(figsize)
#     if title:
#         facet_grid.fig.suptitle(title)
#     if xlabel:
#         facet_grid.set_xlabels(xlabel)
#     if ylabel:
#         facet_grid.set_ylabels(ylabel)
#     if xlim:
#         facet_grid.set(xlim=xlim)
#     if ylim:
#         facet_grid.set(ylim=ylim)

#     return facet_grid


# def barplot(data):
#     fig = sns.barplot(data)
#     return fig


# def histogram(data):
#     fig = sns.histplot(data)
#     return fig


# def boxplot(data):
#     fig = sns.boxplot(data)
#     return fig


# def heatmap(data):
#     fig = sns.heatmap(data, center=1)
#     return fig


# def kernel_density_estimate(data):
#     fig = sns.kdeplot(data)
#     return fig


# # TESTING
# df = np.random.randn(30, 30)
# flights = sns.load_dataset("flights")
# flights_wide = flights.pivot("year", "month", "passengers")
# # print("year" in flights.columns)
# # lineplot(flights, x="passengers")
# penguins = sns.load_dataset("penguins")
# histogram(penguins)
# # lineplot(flights, "year", "passengers")
# # heatmap(df)
# # lineplot(penguins, "flipper_length_mm", "bill_length_mm")

# plt.show()
