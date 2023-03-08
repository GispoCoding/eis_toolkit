import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
from eis_toolkit.exceptions import InvalidDataFrameException, InvalidNumberOfPrincipalComponent

def compute_pca(data_to_compute, num_components):
    """Calculates the pricipal components.

    Args:
        data_to_compute: (dataframe).
        num_components (integer): The  number of principal componets.

    Returns:
        return a dataframe with the principal components values.
        out_meta (dict): The updated metadata.

    Raises:
        InvalidDataFrameException:The dataframe is null.
        InvalidNumberOfPrincipalComponent: The number of principal components should be >= 1.
    """

    if not data_to_compute.shape[1]:
        raise InvalidDataFrameException
    
    if num_components <= 1:
        raise InvalidNumberOfPrincipalComponent

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
    return dataframe

