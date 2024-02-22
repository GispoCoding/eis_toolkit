import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from beartype import beartype
from beartype.typing import Optional, Tuple
from matplotlib.cm import ScalarMappable
from matplotlib.path import Path
from sklearn.preprocessing import LabelEncoder

from eis_toolkit.exceptions import EmptyDataFrameException, InconsistentDataTypesException, InvalidColumnException


def _normalize_data(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    y_min = np.nanmin(data, axis=0)
    y_max = np.nanmax(data, axis=0)
    dy = y_max - y_min
    y_min -= dy * 0.05
    y_max += dy * 0.05
    dy = y_max - y_min

    normalized_data = np.zeros_like(data)
    normalized_data[:, 0] = data[:, 0]
    normalized_data[:, 1:] = (data[:, 1:] - y_min[1:]) / dy[1:] * dy[0] + y_min[0]

    return normalized_data, y_min, y_max


def _get_default_palette(color_data_numeric: bool) -> str:
    if color_data_numeric:
        return "Spectral"
    else:
        return "bright"


@beartype
def _plot_parallel_coordinates(
    data: np.ndarray,
    data_labels: np.ndarray,
    color_data: np.ndarray,
    color_column_name: str,
    plot_title: Optional[str],
    palette_name: Optional[str],
    curved_lines: bool,
) -> matplotlib.figure.Figure:

    fig, main_axis = plt.subplots()

    # If color_data is not numeric, encode the color data to create colors from palette correctly
    if not np.issubdtype(type(color_data[0]), np.number):
        color_data_numeric = False
        label_encoder = LabelEncoder()
        color_data = list(label_encoder.fit_transform(color_data))
    else:
        label_encoder = None
        color_data_numeric = True

    # If palette name is not provided, go with the default
    palette_name = _get_default_palette(color_data_numeric) if not palette_name else palette_name

    # Normalize data for drawing lines
    normalized_data, y_min, y_max = _normalize_data(data)

    # Set style
    axes_list = [main_axis] + [main_axis.twinx() for _ in range(normalized_data.shape[1] - 1)]
    for i, axis in enumerate(axes_list):
        axis.set_ylim(y_min[i], y_max[i])
        axis.spines["top"].set_visible(False)
        axis.spines["bottom"].set_visible(False)
        if axis != main_axis:
            axis.spines["right"].set_visible(False)
            axis.yaxis.set_ticks_position("left")
            axis.spines["left"].set_position(("axes", i / (normalized_data.shape[1] - 1)))

    main_axis.set_xlim(0, normalized_data.shape[1] - 1)
    main_axis.set_xticks(range(normalized_data.shape[1]))
    main_axis.set_xticklabels(data_labels, fontsize=10)
    main_axis.tick_params(axis="x", which="major", pad=7)
    main_axis.spines["right"].set_visible(False)
    main_axis.xaxis.tick_top()
    main_axis.set_title("Parallel Coordinates Plot" if plot_title is None else plot_title, fontsize=18)

    # Create colors
    norm = plt.Normalize(min(color_data), max(color_data))
    cmap = sns.color_palette(palette_name, as_cmap=True)

    if isinstance(cmap, list):
        num_categories = len(set(color_data))
        colors = cmap[:num_categories]  # Take the first N colors
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_colormap", colors)
    elif not color_data_numeric:
        num_categories = len(set(color_data))
        colors = cmap(np.linspace(0, 1, num_categories))
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_colormap", colors)

    if not color_data_numeric:
        # Create the legend for categorical data
        color_boxes = [
            patches.Patch(color=colors[i], label=label_encoder.inverse_transform([i])[0]) for i in range(num_categories)
        ]
        plt.legend(handles=color_boxes, title=color_column_name, loc="best")
    else:
        # Create the colorbar for numerical data
        colorbar_mappable = ScalarMappable(cmap=cmap, norm=norm)
        colorbar_mappable.set_array([])
        # colorbar = plt.colorbar(colorbar_mappable)
        # colorbar.set_label(color_column_name, fontsize=14)

    # Draw lines
    for i in range(data.shape[0]):
        color = cmap(norm(color_data[i]))
        if curved_lines:
            x = np.linspace(0, len(normalized_data) - 1, len(normalized_data) * 3 - 2, endpoint=True)
            y = np.repeat(normalized_data[i, :], 3)[1:-1]

            control_points = list(zip(x, y))
            codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(control_points) - 1)]
            path = Path(control_points, codes)

            curve_patch = patches.PathPatch(path, facecolor="none", edgecolor=color, lw=1, alpha=0.5)
            main_axis.add_patch(curve_patch)
        else:
            main_axis.plot(range(normalized_data.shape[1]), normalized_data[i, :], c=color, lw=1, alpha=0.5)

    plt.tight_layout()
    return fig


@beartype
def plot_parallel_coordinates(
    df: pd.DataFrame,
    color_column_name: str,
    plot_title: Optional[str] = None,
    palette_name: Optional[str] = None,
    curved_lines: bool = True,
) -> matplotlib.figure.Figure:
    """Plot a parallel coordinates plot.

    Automatically removes all rows containing null/nan values. Tries to convert columns to numeric
    to be able to plot them. If more than 8 columns are present (after numeric filtering), keeps only
    the first 8 to plot.

    Args:
        df: The DataFrame to plot.
        color_column_name: The name of the column in df to use for color encoding.
        plot_title: The title for the plot. Default is None.
        palette_name: The name of the color palette to use. Default is None.
        curved_lines: If True, the plot will have curved instead of straight lines. Default is True.

    Returns:
        A matplotlib figure containing the parallel coordinates plot.

    Raises:
        EmptyDataFrameException: Raised when the DataFrame is empty.
        InvalidColumnException: Raised when the color column is not found in the DataFrame.
        InconsistentDataTypesException: Raised when the color column has multiple data types.
    """

    if df.empty:
        raise EmptyDataFrameException("The input DataFrame is empty.")

    if color_column_name not in df.columns:
        raise InvalidColumnException(f"The provided color column {color_column_name} is not found in the DataFrame.")

    df = df.convert_dtypes()
    df = df.apply(pd.to_numeric, errors="ignore")

    color_data = df[color_column_name].to_numpy()
    if len(set([type(elem) for elem in color_data])) != 1:
        raise InconsistentDataTypesException(
            "The color column should have a consistent datatype. Multiple data types detected in the color column."
        )

    df = df.select_dtypes(include=np.number)

    # Drop non-numeric columns and the column used for coloring
    columns_to_drop = [color_column_name]
    for column in df.columns.values:
        if df[column].isnull().all():
            columns_to_drop.append(column)
    df = df.loc[:, ~df.columns.isin(columns_to_drop)]

    # Keep only first 8 columns if more are still present
    if len(df.columns.values) > 8:
        df = df.iloc[:, :8]

    data_labels = df.columns.values
    data = df.to_numpy()

    fig = _plot_parallel_coordinates(
        data=data,
        data_labels=data_labels,
        color_data=color_data,
        color_column_name=color_column_name,
        plot_title=plot_title,
        palette_name=palette_name,
        curved_lines=curved_lines,
    )
    return fig
