import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt


# Reading the data

num_components = 2
#importing 1112X9 data frame
path1 = "1112x9.csv"
#importing 17X9 data frame
path2 = "17x9.csv"
df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)

#defining the function for PCA

def MyPCA(X):
    # standardize the Data
    features1 = X.columns

    # Separating out the features
    x1 = X.loc[:, features1].values

    # Standardizing the features
    x1 = StandardScaler().fit_transform(x1)

    # Calculating the PCA
    pca = PCA(n_components=num_components)
    principalComponents1 = pca.fit_transform(x1)
    dataframe = pd.DataFrame(data=principalComponents1, columns=['principal_component_1', 'principal_component_2'])

    return dataframe


# Defining the function for plotting the PCA

def PCA_plot(X1, X2):
    #plotting the graph
    plt.scatter(X2.principal_component_1, X2.principal_component_2,marker='.', color='blue', s=200, alpha = .2)
    plt.scatter(X1.principal_component_1, X1.principal_component_2,marker='*', color='red', s=500)
    matplotlib.rcParams.update({'font.size': 22})
    plt.title("Principal components of all points")
    plt.xlabel('Principal component 1')
    plt.ylabel('Principal component 2')

    plt.show()
    
    
# Calling the function for PCA

principal_df1 = MyPCA(X=df1)
principal_df2 = MyPCA(X=df2)

#calling the function for scatter plot

PCA_plot(principal_df2,principal_df1)