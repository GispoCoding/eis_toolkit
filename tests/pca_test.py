import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from eis_toolkit.exploratory_analyses.pca import _compute_pca


def plot_pca(X1: pd.Dataframe, X2: pd.Dataframe) -> None:
    """Do the pricipal components.

    Args:
        X1: (dataframe).
        X2: (dataframe).

    Returns:
        return None.
    """

    # plotting the graph
    plt.scatter(X2.principal_component_1, X2.principal_component_2, marker=".", color="blue", s=200, alpha=0.2)
    plt.scatter(X1.principal_component_1, X1.principal_component_2, marker="*", color="red", s=500)
    matplotlib.rcParams.update({"font.size": 22})
    plt.title("Principal components of all points")
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")

    plt.savefig("test_pca.png")


num_components = 2

# do the first PCA
data_to_analyse_1 = pd.read_csv("data/local/data/17x9.csv")
returned_principal_components_a = _compute_pca(data_to_analyse_1, num_components)

# DO THE SECOND PCA
data_to_analyse_2 = pd.read_csv("data/local/data/1112x9.csv")
returned_principal_components_b = _compute_pca(data_to_analyse_2, num_components)

plot_pca(returned_principal_components_a, returned_principal_components_b)
