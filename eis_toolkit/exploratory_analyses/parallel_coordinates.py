import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from beartype.typing import Sequence, Tuple
from matplotlib.path import Path
from sklearn.preprocessing import LabelEncoder


class ParallelCoordinatesPlot:
    """Class for parallel coordinates plot using matplotlib."""

    def __init__(
        self, data: np.ndarray, categories: Sequence[int], variable_names: Sequence[str], draw_curves: bool = True
    ):
        """Init method."""
        self.data = data
        self.categories = categories
        self.variable_names = variable_names
        self.draw_curves = draw_curves
        self.num_features = data.shape[0]
        self.fig, self.main_axis = plt.subplots()

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, category_column: str, draw_curves=True):
        """Create ParallelCoordinatesPlot from a dataframe."""
        categories = df[category_column]
        label_encoder = LabelEncoder()
        encoded_categories = label_encoder.fit_transform(categories)

        data = df.drop(columns=[category_column])
        return cls(data.to_numpy(), encoded_categories, data.columns.values, draw_curves)

    def plot(self):
        """Plot the graph."""
        normalized_data, y_min, y_max = self._normalize_data()

        self._set_style(y_min, y_max)

        self._draw_lines(normalized_data)

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
        self.main_axis.set_xticklabels(self.variable_names, fontsize=14)
        self.main_axis.tick_params(axis="x", which="major", pad=7)
        self.main_axis.spines["right"].set_visible(False)
        self.main_axis.xaxis.tick_top()
        self.main_axis.set_title("Parallel Coordinates Plot", fontsize=18)

    def _draw_lines(self, normalized_data: np.ndarray) -> None:
        # colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(self.categories))))
        # colors = plt.cm.tab10.colors
        colors = sns.color_palette("bright", self.num_features)

        for i in range(self.num_features):
            if self.draw_curves:
                x = np.linspace(0, len(normalized_data) - 1, len(normalized_data) * 3 - 2, endpoint=True)
                y = np.repeat(normalized_data[i, :], 3)[1:-1]
                control_points = list(zip(x, y))
                codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(control_points) - 1)]
                path = Path(control_points, codes)
                curve_patch = patches.PathPatch(
                    path, facecolor="none", lw=1, edgecolor=colors[self.categories[i]], alpha=0.5
                )
                self.main_axis.add_patch(curve_patch)
            else:
                self.main_axis.plot(range(self.data.shape[1]), normalized_data[i, :], c=colors[self.categories[i]])

    # def draw_lines(self, host, data, num_instances, categories, axis_labels, use_curves=True):
    #     """Draws lines for each instance on the plot."""
    #     colors = plt.cm.tab10.colors
    #     for i in range(num_instances):
    #         if use_curves:
    #             # create bezier curves
    #             x = np.linspace(0, len(data) - 1, len(data) * 3 - 2, endpoint=True)
    #             y = np.repeat(data[i, :], 3)[1:-1]
    #             control_points = list(zip(x, y))
    #             codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(control_points) - 1)]
    #             path = Path(control_points, codes)
    #             curve_patch = patches.PathPatch(
    #                 path, facecolor='none', lw=1, edgecolor=colors[categories[i] - 1], alpha=0.5
    #             )
    #             host.add_patch(curve_patch)
    #         else:
    #             host.plot(range(data.shape[1]), data[i,:], c=colors[(categories[i]) % len(colors) ])


# def plot_parallel_coordinates_from_df(df: pd.DataFrame, category_column: str, curved = True):
#     categories = df[category_column].to_numpy()
#     label_encoder = LabelEncoder()
#     numerical_categories = list(label_encoder.fit_transform(categories))

#     data = df.loc[:, df.columns != category_column]
#     data_labels = data.columns.values
#     data_arr = data.to_numpy()
#     number_of_features = data.shape[0]
#     plot_parallel_coordinates(data_arr, numerical_categories, data_labels, number_of_features, draw_curves = curved)


# def plot_parallel_coordinates_dummy():
#     ynames = ['P1', 'P2', 'P3', 'P4', 'P5']
#     N1, N2, N3 = 10, 5, 8
#     N = N1 + N2 + N3
#     category = np.concatenate([np.full(N1, 1), np.full(N2, 2), np.full(N3, 3)])
#     y1 = np.random.uniform(0, 10, N) + 7 * category
#     y2 = np.sin(np.random.uniform(0, np.pi, N)) ** category
#     y3 = np.random.binomial(300, 1 - category / 10, N)
#     y4 = np.random.binomial(200, (category / 6) ** 1/3, N)
#     y5 = np.random.uniform(0, 800, N)
#     # organize the data
#     ys = np.dstack([y1, y2, y3, y4, y5])[0]
#     plot_parallel_coordinates(ys, category, ynames, N)


# PLOT IRIS DATA FROM DF
iris = sns.load_dataset("iris")
category_column = "species"

pl_plot = ParallelCoordinatesPlot.from_dataframe(iris, category_column)
pl_plot.plot()

# PLOT DUMMY DATA
# plot_parallel_coordinates_dummy()
