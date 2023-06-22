from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from beartype.typing import Sequence, Tuple
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.path import Path
from sklearn.preprocessing import LabelEncoder


class ParallelCoordinatesPlot:
    """Class for parallel coordinates plot using matplotlib."""

    def __init__(
        self,
        data: np.ndarray,
        data_labels: np.ndarray,
        color_data: np.ndarray,
        color_data_label: str,
        palette_name: Optional[str],
        draw_curves: bool = True,
    ):
        """Init method."""
        self.data = data
        self.data_labels = data_labels
        self.color_data = color_data
        self.color_data_label = color_data_label
        self.draw_curves = draw_curves

        # TODO: Add NaN check or check numerics?
        if len(set([type(elem) for elem in color_data])) != 1:
            raise Exception("Data used for coloring has multiple data types")

        # If category_column is not numeric
        # TODO: Add checking that labels can be transformed
        if not np.issubdtype(type(color_data[0]), np.number):
            self.color_data_numeric = False
            self.label_encoder = LabelEncoder()
            self.color_data = list(self.label_encoder.fit_transform(color_data))
        else:
            self.label_encoder = None
            self.color_data_numeric = True

        # If palette name is not provided, go with default
        self.palette_name = self.get_default_palette() if not palette_name else palette_name

        self.num_features = data.shape[0]

        self.fig, self.main_axis = plt.subplots()

    def get_default_palette(self):
        if self.color_data_numeric:
            return "Spectral"
        else:
            return "bright"

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, color_column_name: str, palette_name: Optional[str] = None, draw_curves=True
    ):
        """Create ParallelCoordinatesPlot from a dataframe."""

        color_data = df[color_column_name].to_numpy()

        columns_to_drop = [color_column_name]
        for column in df.columns.values:
            if not np.issubdtype(df[column].dtype, np.number):
                columns_to_drop.append(column)
                print("Dropped column: ", column)

        filtered_df = df.loc[:, ~df.columns.isin(columns_to_drop)]
        data_labels = filtered_df.columns.values

        data = filtered_df.to_numpy()

        return cls(data, data_labels, color_data, color_column_name, palette_name, draw_curves)

    def plot(self):
        """Plot the graph."""
        normalized_data, y_min, y_max = self._normalize_data()

        self._set_style(y_min, y_max)

        self._draw_lines(normalized_data)

        if self.color_data_numeric:
            colorbar = plt.colorbar(self.colorbar_mappable)
            colorbar.set_label(self.color_data_label, fontsize=14)

        else:  # TODO: Couple color data numeric and discerete colors
            plt.legend(handles=self.color_boxes)

        plt.tight_layout()
        plt.show()

    def _normalize_data(self) -> Tuple[np.ndarray, float, float]:
        y_min = self.data.min(axis=0)
        y_max = self.data.max(axis=0)
        dy = y_max - y_min
        y_min -= dy * 0.05
        y_max += dy * 0.05
        dy = y_max - y_min

        normalized_data = np.zeros_like(self.data)
        normalized_data[:, 0] = self.data[:, 0]
        normalized_data[:, 1:] = (self.data[:, 1:] - y_min[1:]) / dy[1:] * dy[0] + y_min[0]

        return normalized_data, y_min, y_max

    def _set_style(self, y_min: float, y_max: float) -> None:
        axes_list = [self.main_axis] + [self.main_axis.twinx() for _ in range(self.data.shape[1] - 1)]
        for i, axis in enumerate(axes_list):
            axis.set_ylim(y_min[i], y_max[i])
            axis.spines["top"].set_visible(False)
            axis.spines["bottom"].set_visible(False)
            if axis != self.main_axis:
                axis.spines["right"].set_visible(False)
                axis.yaxis.set_ticks_position("left")
                axis.spines["left"].set_position(("axes", i / (self.data.shape[1] - 1)))

        self.main_axis.set_xlim(0, self.data.shape[1] - 1)
        self.main_axis.set_xticks(range(self.data.shape[1]))
        self.main_axis.set_xticklabels(self.data_labels, fontsize=14)
        self.main_axis.tick_params(axis="x", which="major", pad=7)
        self.main_axis.spines["right"].set_visible(False)
        self.main_axis.xaxis.tick_top()
        self.main_axis.set_title("Parallel Coordinates Plot", fontsize=18)

    # def _create_colormap(self, colors, name='custom_colormap', num_categories=None):
    #     """Create a colormap from seaborn color palette."""
    #     if not self.numerical_categories and num_categories <= len(colors):
    #         colors = colors[:num_categories]  # Take the first N colors
    #     cmap = mcolors.LinearSegmentedColormap.from_list(name, colors)
    #     return cmap

    def _draw_lines(self, data: np.ndarray) -> None:
        self.norm = plt.Normalize(min(self.color_data), max(self.color_data))
        cmap = sns.color_palette(self.palette_name, as_cmap=True)

        # If seaborn did not produce a cmap
        if type(cmap) == list:
            num_categories = len(set(self.color_data))
            # i can be larger than len(cmap) ?
            self.color_boxes = [
                patches.Patch(color=cmap[i], label=self.label_encoder.inverse_transform([i])[0])
                for i in range(num_categories)
            ]
            if not self.color_data_numeric and num_categories <= len(cmap):
                colors = cmap[:num_categories]  # Take the first N colors
            cmap = mcolors.LinearSegmentedColormap.from_list("custom_colormap", colors)
            # cmap = self._create_colormap(cmap, num_categories=num_categories)
        else:
            self.colorbar_mappable = ScalarMappable(cmap=cmap, norm=self.norm)
            self.colorbar_mappable.set_array([])
        self.cmap = cmap

        for i in range(self.num_features):
            color = cmap(self.norm(self.color_data[i]))
            if self.draw_curves:
                x = np.linspace(0, len(data) - 1, len(data) * 3 - 2, endpoint=True)
                y = np.repeat(data[i, :], 3)[1:-1]
                control_points = list(zip(x, y))
                codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(control_points) - 1)]
                path = Path(control_points, codes)
                curve_patch = patches.PathPatch(path, facecolor="none", lw=1, edgecolor=color, alpha=0.5)
                self.main_axis.add_patch(curve_patch)
            else:
                self.main_axis.plot(range(self.data.shape[1]), data[i, :], c=color)


# PLOT IRIS DATA FROM DF
iris = sns.load_dataset("iris")
color_column = "species"

pl_plot = ParallelCoordinatesPlot.from_dataframe(iris, color_column)
pl_plot.plot()
