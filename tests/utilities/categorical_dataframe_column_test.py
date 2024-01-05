import pandas as pd
import numpy as np

from eis_toolkit.utilities.checks.dataframe import check_columns_categorical

ints_and_bools = pd.DataFrame({"e": [0, 0, 1, 1], "f": [True, False, True, True]})
floats = pd.DataFrame({"c": [0.1, 0.5, 0.3, 0.2], "d": [1.0, 2.5, 3.0, 4.0]})
too_many_uniques = pd.DataFrame({
    "a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "b": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
})


def test_ints_and_bools():
    """Test that booleans and max_unique_values > number of unique ints returns True."""
    result = check_columns_categorical(ints_and_bools, ints_and_bools.columns.to_list())
    assert result is True


def test_floats():
    """Test that max_unique_values > number of unique floats returns True."""
    assert check_columns_categorical(floats, floats.columns.to_list()) is True


def test_too_many_uniques():
    """Test that too many unique ints and floats returns False."""
    result = check_columns_categorical(too_many_uniques, columns=["a"])
    assert result is False
    result = check_columns_categorical(too_many_uniques, columns=["b"])
    assert result is False


def test_max_uniques_value():
    """Test that adjusted max_unique_values < number of unique ints and floats returns False."""
    assert check_columns_categorical(floats, columns=["c"], max_unique_values=3) is False
    assert check_columns_categorical(ints_and_bools, columns=["e"], max_unique_values=1) is False


def test_unique_values():
    """Test that adjusted unique values == number of unique ints returns True."""
    assert check_columns_categorical(ints_and_bools, columns=["e"], max_unique_values=2) is True
