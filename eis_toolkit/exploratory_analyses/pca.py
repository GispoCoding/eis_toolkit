from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidNumberOfPrincipalComponents


def _compute_pca(  # type: ignore[no-any-unimported]
    data: pd.DataFrame, n_components: int
) -> Tuple[pd.DataFrame, np.ndarray]:

    feature_matrix = data.loc[:, data.columns].values
    standardized_data = StandardScaler().fit_transform(feature_matrix)

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(standardized_data)
    explained_variances = pca.explained_variance_ratio_

    component_names = [f"principal_component_{i+1}" for i in range(n_components)]

    principal_component_df = pd.DataFrame(data=principal_components, columns=component_names)
    return principal_component_df, explained_variances


def compute_pca(  # type: ignore[no-any-unimported]
    data: pd.DataFrame, n_components: int
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Compute the principal components for the input data.

    Args:
        data: A DataFrame containing the input data.
        n_components: The number of principal components to compute.

    Returns:
        DataFrame containing the computed principal components.
        The explained variance ratios for each component.

    Raises:
        EmptyDataFrameException: Raised when the input DataFrame is empty.
        InvalidNumberOfPrincipalComponents: Raised when the number of principal components is less than 2.
    """

    if data.empty:
        raise EmptyDataFrameException("The input DataFrame is empty.")

    if n_components < 2 or n_components > len(data.columns):
        raise InvalidNumberOfPrincipalComponents(
            "The number of principal components should be >= 2 and at most the number of columns in the DataFrame."
        )

    principal_component_df, explained_variances = _compute_pca(data, n_components)
    return principal_component_df, explained_variances
