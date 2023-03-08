import pandas as pd 
import numpy as np
from eis_toolkit.exploratory_analyses.pca import compute_pca 
import matplotlib.pyplot as plt 
import matplotlib

# Defining the function for plotting the PCA
def pca_plot_hardcoded_to_remove_later(X1, X2):
    #plotting the graph
    plt.scatter(X2.principal_component_1, X2.principal_component_2,marker='.', color='blue', s=200, alpha = .2)
    plt.scatter(X1.principal_component_1, X1.principal_component_2,marker='*', color='red', s=500)
    matplotlib.rcParams.update({'font.size': 22})
    plt.title("Principal components of all points")
    plt.xlabel('Principal component 1')
    plt.ylabel('Principal component 2')

    plt.show()

num_components = 2

# do the first PCA
data_to_analyse_1 = pd.read_csv("/home/dipak/Documents/eis_toolkit/tests/data/local/data/1112x9.csv")
returned_principal_components_a = compute_pca(data_to_analyse_1, num_components)

# DO THE SECOND PCA 
data_to_analyse_2 = pd.read_csv("/home/dipak/Documents/eis_toolkit/tests/data/local/data/17x9.csv")
returned_principal_components_b = compute_pca(data_to_analyse_2, num_components)

pca_plot_hardcoded_to_remove_later(returned_principal_components_a, returned_principal_components_b)
