import numpy as np
import pandas as pd
import pytest
import scipy

from eis_toolkit.transformations.one_hot_encoding import one_hot_encode


@pytest.fixture
def sample_dataframe():
    """Return sample DataFrame for testing."""
    return pd.DataFrame({"A": ["cat", "dog", "fish"], "B": ["apple", "banana", "orange"], "C": [1, 2, 3]})


@pytest.fixture
def sample_numpy_array(sample_dataframe):
    """Return sample Numpy array made from the sample DataFrame."""
    return sample_dataframe.to_numpy()


def test_encode_dataframe_sparse_all_columns(sample_dataframe):
    """Test that encoding DataFrame with sparse output works as expected."""
    encoded_df = one_hot_encode(sample_dataframe)
    assert all(item in encoded_df.columns for item in ["A_cat", "A_dog", "A_fish", "B_apple", "B_banana", "B_orange"])
    assert encoded_df.dtypes.apply(pd.api.types.is_sparse).all()


def test_encode_dataframe_sparse_selected_columns(sample_dataframe):
    """Test that encoding DataFrame with sparse output with column selections works as expected."""
    encoded_df = one_hot_encode(sample_dataframe, columns=["A", "B"])
    encoded_df_without_column_C = encoded_df.drop(["C"], axis=1)
    assert "C" in encoded_df.columns
    assert "A_fish" in encoded_df.columns
    assert encoded_df_without_column_C.dtypes.apply(pd.api.types.is_sparse).all()
    assert encoded_df["C"].dtype == int


def test_encode_dataframe_dense(sample_dataframe):
    """Test that encoding DataFrame with dense output works as expected."""
    encoded_df = one_hot_encode(sample_dataframe, sparse_output=False)
    assert not encoded_df.dtypes.apply(pd.api.types.is_sparse).any()


def test_encode_numpy_array_sparse(sample_numpy_array):
    """Test that encoding Numpy array with sparse output works as expected."""
    encoded_data = one_hot_encode(sample_numpy_array)
    assert isinstance(encoded_data, scipy.sparse._csr.csr_matrix)


def test_encode_numpy_array_dense(sample_numpy_array):
    """Test that encoding Numpy array with dense output works as expected."""
    encoded_data = one_hot_encode(sample_numpy_array, sparse_output=False)
    assert isinstance(encoded_data, np.ndarray)
    assert len(encoded_data[0]) > len(sample_numpy_array[0])
