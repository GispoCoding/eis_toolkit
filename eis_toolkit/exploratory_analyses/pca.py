import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from eis_toolkit.exceptions import InvalidDataFrameException, InvalidNumberOfPrincipalComponent


def _compute_pca(data_to_compute: pd.DataFrame, num_components: int) -> pd.DataFrame:
    """Do the pricipal components.

    Args:
        data_to_compute: (dataframe).
        num_components (integer).

    Returns:
        return a dataframe with the principal components values.

    Raises:
        InvalidDataFrameException: The dataframe is null.
        InvalidNumberOfPrincipalComponent: The number of principal components should be >= 2.
    """

    if not data_to_compute.shape[1]:
        raise InvalidDataFrameException

    if num_components <= 1:
        raise InvalidNumberOfPrincipalComponent

    data = data_to_compute.loc[:, data_to_compute.columns].values

    data = StandardScaler().fit_transform(data)

    pca = PCA(n_components=num_components)
    principalComponent = pca.fit_transform(data)

    columns_name_holder = []
    for i in range(num_components):
        columns_name_holder.append(f"principal_component_{i+1}")

    dataframe = pd.DataFrame(data=principalComponent, columns=columns_name_holder)
    return dataframe
