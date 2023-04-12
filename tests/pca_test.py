import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidNumberOfPrincipalComponents
from eis_toolkit.exploratory_analyses.pca import compute_pca

data = pd.DataFrame({"A": [1, 2, 3], "B": [1, 2, 3]})


def test_pca_output():
    """Test that PCA function gives output in intended format."""
    pca_df, explained_variances = compute_pca(data, 2)
    expected_columns = ["principal_component_1", "principal_component_2"]

    assert explained_variances.size == 2
    assert list(pca_df.columns) == expected_columns
    assert pca_df.shape == data.shape


def test_pca_values():
    """Test that PCA function returns correct output values."""
    pca_df, explained_variances = compute_pca(data, 2)
    expected_pca_values = np.array([[-1.73205081, 1.11022302e-16], [0.0, 0.0], [1.73205081, 1.11022302e-16]])
    expected_explained_variances_values = [1.0, 4.10865055e-33]

    np.testing.assert_array_almost_equal(pca_df.values, expected_pca_values, decimal=8)
    np.testing.assert_array_almost_equal(explained_variances, expected_explained_variances_values, decimal=8)


def test_empty_dataframe():
    """Test that empty dataframe raises the correct exception."""
    empty_df = pd.DataFrame()
    with pytest.raises(EmptyDataFrameException):
        compute_pca(empty_df, 2)


def test_invalid_number_of_components():
    """Test that invalid number of PCA components raises the correct exception."""
    with pytest.raises(InvalidNumberOfPrincipalComponents):
        compute_pca(data, 1)
