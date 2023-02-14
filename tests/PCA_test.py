import pandas as pd 
import numpy as np
from PCA import * 


# Defining the function for plotting the PCA
def PCA_plot_HARDCODED_TO_REMOVE(X1, X2):
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
data_to_analyse_1 = pd.read_csv("data/local/data/pca_1112x9.csv")
returned_principal_components = Compute_PCA(data_to_analyse_1, num_components, show_results=True)

# DO THE SECOND PCA 
data_to_analyse_2 = pd.read_csv("data/local/data/pca_17x9.csv")
returned_principal_components = Compute_PCA(data_to_analyse_2, num_components, show_results=True)