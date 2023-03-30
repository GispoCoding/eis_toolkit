from typing import Any, Dict, List, Optional

import numpy as np
from matplotlib import pyplot as plt


def _histogram(data_array: np.ndarray, column_index: int, **kwargs: Dict[str, List[int]]) -> Any:
    selected_column = data_array[:, int(column_index)]
    histogram = plt.hist(selected_column, **kwargs)
    return histogram


def histogram(data_array: np.ndarray, column_index: int, **kwargs: Dict[str, List[int]]) -> Any:
    """Produce histogram from input data.

    This function acts a wrapper for the matplotlib histogram function, for further info see matplotlib.pyplot.hist().
    """
    histogram = _histogram(data_array=data_array, column_index=column_index, **kwargs)
    return histogram


def _scatterplot(
    data_array: np.ndarray, x_column_index: int, y_column_index: int, **kwargs: Dict[str, List[int]]
) -> Any:
    x_col = data_array[:, int(x_column_index)]
    y_col = data_array[:, int(y_column_index)]
    scatter = plt.scatter(x_col, y_col, **kwargs)
    return scatter


def scatterplot(
    data_array: np.ndarray, x_column_index: int, y_column_index: int, **kwargs: Dict[str, List[int]]
) -> Any:
    """Produce scatterplot from input data.

    This function acts a wrapper for the matplotlib scatterplot function, for further info see matplotlib.pyplot.hist().
    """
    scatter = _scatterplot(
        data_array=data_array, x_column_index=x_column_index, y_column_index=y_column_index, **kwargs
    )
    return scatter


def _boxplot(data_array: np.ndarray, column_index: Optional[int] = None, **kwargs: Dict[str, List[int]]) -> Any:

    # NaN values are filtered out, as the boxplot returns empty if there is even a single NaN value.
    data_array = data_array[~np.isnan(data_array).any(axis=1), :]
    if column_index is not None:
        plot_data = data_array[:, int(column_index)]
    else:
        plot_data = data_array

    box = plt.boxplot(plot_data, **kwargs)
    return box


def boxplot(data_array: np.ndarray, column_index: Optional[int] = None, **kwargs: Dict[str, List[int]]) -> Any:
    """Produce boxplot from input data.

    This function acts a wrapper for the matplotlib boxplot function, for further info see matplotlib pyplot boxplot().
    Accepts input as ndarray, and if the optional column_index parameter is provided,
    selects a single data column from the input data to plot.
    """
    box = _boxplot(data_array=data_array, column_index=column_index, **kwargs)
    return box
