import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import InvalidColumnIndexException
from eis_toolkit.utilities.miscellaneous import (
    rename_columns,
    rename_columns_by_pattern,
    replace_values,
    replace_values_df,
)


def test_replace_values_1():
    """Test that replacing specified values in a Numpy array works as expected. Case 1."""
    data = np.array([[1, 2, 3, 2, 1], [2, 3, 5, 4, 3]])
    target_arr = np.array([[1, 5555, 5555, 5555, 1], [5555, 5555, 5, 4, 5555]])
    replaced_arr = replace_values(data, values_to_replace=[2, 3], replace_value=5555)
    assert np.array_equal(replaced_arr, target_arr)


def test_replace_values_2():
    """Test that replacing specified values in a Numpy array works as expected. Case 2."""
    data = np.array([[1, 2, 3, 2, 1], [2, 3, 5, 4, 3]])
    target_arr = np.array([[np.nan, 2, 3, 2, np.nan], [2, 3, 5, 4, 3]])
    replaced_arr = replace_values(data, values_to_replace=1, replace_value=np.nan)
    assert np.allclose(replaced_arr, target_arr, equal_nan=True)


def test_replace_values_df():
    """Test that replacing specified values in a DataFrame works as expected."""
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [3, 5, 6]})
    target_df = pd.DataFrame({"col1": [1, 2, np.nan], "col2": [3, 5, 6]})
    replaced_df = replace_values_df(df, values_to_replace=3, replace_value=np.nan, columns=["col1"])
    assert replaced_df.equals(target_df)


def test_rename_columns_by_pattern():
    """Test that renaming columns of a DataFrame with a given pattern works as expected."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    target_df = pd.DataFrame({"V1": [1, 2, 3], "V2": [4, 5, 6]})
    renamed_df = rename_columns_by_pattern(df)
    pd.testing.assert_frame_equal(renamed_df, target_df)

    target_df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    renamed_df = rename_columns_by_pattern(df, pattern="col")
    pd.testing.assert_frame_equal(renamed_df, target_df)


def test_rename_columns_with_too_many_columns():
    """Test that renaming columns of a DataFrame works as expected."""
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4], "col3": [5, 6]})
    with pytest.raises(InvalidColumnIndexException):
        rename_columns(df, ["a", "b", "c", "d"])


def test_rename_columns():
    """Test that renaming columns of a DataFrame works as expected."""
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4], "col3": [5, 6]})
    colnames = ["a", "b", "c"]
    target_df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    renamed_df = rename_columns(df, colnames=colnames)
    pd.testing.assert_frame_equal(renamed_df, target_df)
