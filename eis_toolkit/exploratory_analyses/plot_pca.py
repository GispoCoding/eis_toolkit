import numpy as np
import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure


def plot_pca(  # type: ignore[no-any-unimported]
    pca_df: pd.DataFrame, explained_variances: np.ndarray, color_feat: pd.Series = None, save_path: str = ""
) -> Figure:
    """Plot a scatter matrix of different principal component combinations.

    Args:
        pca_df: A DataFrame containing the principal components.
        explained_variances: The explained variance ratios for each principal component.
        color_feat: Feature in the original data that was not used for PCA. Categorical data
            that can be used for coloring points in the plot. Optional parameter.
        save_path: The save path for the plot. If empty, no saving

    Returns:
        The plotly figure object.
    """

    labels = {str(i): f"PC {i+1} ({var:.1f}%)" for i, var in enumerate(explained_variances * 100)}

    fig = px.scatter_matrix(
        pca_df.to_numpy(), labels=labels, dimensions=range(explained_variances.size), color=color_feat
    )
    fig.update_traces(diagonal_visible=False)

    if save_path != "":
        fig.write_html(save_path)
    fig.show()
    return fig
