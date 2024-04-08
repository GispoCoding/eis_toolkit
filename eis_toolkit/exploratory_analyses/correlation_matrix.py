import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from beartype import beartype
from beartype.typing import Literal, Optional, Sequence

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidParameterValueException, NonNumericDataException
from eis_toolkit.utilities.checks.dataframe import check_empty_dataframe


@beartype
def correlation_matrix(
    data: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    correlation_method: Literal["pearson", "kendall", "spearman"] = "pearson",
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """Compute correlation matrix on the input data.

    It is assumed that the data is numeric, i.e. integers or floats. NaN values are excluded from the calculations.

    Args:
        data: Dataframe containing the input data.
        columns: Columns to include in the correlation matrix. If None, all numeric columns are used.
        correlation_method: 'pearson', 'kendall', or 'spearman'. Defaults to 'pearson'.
        min_periods: Minimum number of observations required per pair of columns to have valid result. Optional.

    Returns:
        Dataframe containing matrix representing the correlation coefficient \
            between the corresponding pair of variables.

    Raises:
        EmptyDataFrameException: The input Dataframe is empty.
        InvalidParameterValueException: min_periods argument is used with method 'kendall'.
        NonNumericDataException: The selected columns contain non-numeric data.
    """
    if check_empty_dataframe(data):
        raise EmptyDataFrameException("The input Dataframe is empty.")

    if columns:
        invalid_columns = [column for column in columns if column not in data.columns]
        if invalid_columns:
            raise InvalidParameterValueException(f"Invalid columns: {invalid_columns}")
        data_subset = data[columns]
    else:
        data_subset = data.select_dtypes(include=np.number)

    if not all(data_subset.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        raise NonNumericDataException("The input data contain non-numeric data.")

    if correlation_method == "kendall" and min_periods is not None:
        raise InvalidParameterValueException(
            "The argument min_periods is available only with correlation methods 'pearson' and 'spearman'."
        )

    matrix = data_subset.corr(method=correlation_method, min_periods=min_periods, numeric_only=True)

    return matrix


@beartype
def plot_correlation_matrix(
    matrix: pd.DataFrame,
    annotate: bool = True,
    cmap: Optional[str] = None,
    plot_title: Optional[str] = None,
    **kwargs: dict,
) -> plt.Axes:
    """
    Create a Seaborn heatmap to visualize correlation matrix.

    Args:
        matrix: Correlation matrix as a DataFrame.
        annotate: If plot squares should display the correlation values. Defaults to True.
        cmap: Colormap name for plotting. Optional parameter. Defaults to None, in which
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
        **kwargs,
    )
    if plot_title is not None:
        ax.set_title(plot_title)

    return ax
