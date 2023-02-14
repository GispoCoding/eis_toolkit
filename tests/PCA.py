import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt

#defining the function for PCA
def Compute_PCA(data_to_compute, num_components, show_results=False):
    """THIS FUNCTOIN TAKES DATAFRAME AS INPUT, CALCULATES THE PRINCIPLE COMPONENTS AND RETURS A DATAFRAME CONTAINING THE PRINCIPAL COMPONETS INSIDE IT"""
    try:
        # Separating out the features
        data = data_to_compute.loc[:, data_to_compute.columns].values

        # Standardizing the features
        data = StandardScaler().fit_transform(data)

        # Calculating the PCA
        pca = PCA(n_components=num_components)
        principalComponent = pca.fit_transform(data)

        columns_name_holder = []
        for i in range(num_components):
            columns_name_holder.append(f'principal_component_{i+1}')

        # create a data frame with ocs inside 
        dataframe = pd.DataFrame(data=principalComponent, columns=columns_name_holder)

        # if show the res is true print it on screen 
        if(show_results):
            print(dataframe)

        return dataframe
    except Exception as e:
        print(f"[EXCEPTION] Compute the pca throws exception {e}")
        return None



    
