import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import InvalidColumnException, InvalidParameterValueException
from eis_toolkit.transformations.coda.pairwise import pairwise_logratio, single_pairwise_logratio


def test_single_pairwise_logratio():
    """Test the pairwise logratio transform core functionality."""
    assert single_pairwise_logratio(1.0, 1.0) == 0
    assert single_pairwise_logratio(80.0, 15.0) == pytest.approx(1.67, abs=1e-2)


def test_single_pairwise_logratio_with_zeros():
    """Test that calling the function with a zero value as either value raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        single_pairwise_logratio(0.0, 1.0)

    with pytest.raises(InvalidParameterValueException):
        single_pairwise_logratio(1.0, 0.0)

    with pytest.raises(InvalidParameterValueException):
        single_pairwise_logratio(0.0, 0.0)


def test_pairwise_logratio():
    """Test the pairwise logratio transform core functionality."""
    arr = np.array([[80, 15, 5], [75, 18, 7]])
    df = pd.DataFrame(arr, columns=["a", "b", "c"])
    result = pairwise_logratio(df, "a", "b")
    assert result[0] == pytest.approx(1.67, abs=1e-2)
    assert result[1] == pytest.approx(1.43, abs=1e-2)


def test_pairwise_logratio_with_unexpected_column_name():
    """Test that providing an invalid column name raises the correct exception."""
    with pytest.raises(InvalidColumnException):
        arr = np.array([[65, 12, 18, 5], [63, 16, 15, 6]])
        df = pd.DataFrame(arr, columns=["a", "b", "c", "d"])
        pairwise_logratio(df, "x", "y")


def test_pairwise_logratio_with_zero_values():
    """Test that providing a column with zero values raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        arr = np.array([[65, 23, 0, 12], [63, 21, 1, 17]])
        df = pd.DataFrame(arr, columns=["a", "b", "c", "d"])
        pairwise_logratio(df, "b", "c")
