import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import InvalidColumnException, NumericValueSignException
from eis_toolkit.transformations.coda.ilr import _calculate_scaling_factor, single_ILR_transform


def test_calculate_scaling_factor():
    """Test the scaling factor calculation."""
    result = _calculate_scaling_factor(2, 1)
    expected = np.sqrt(2 / 3.0)  # 0.816496580927726
    assert result == expected


def test_single_ILR_transform_with_single_composition():
    """Test the core functionality of a single ILR transform with a single row of data."""
    arr = np.array([80, 15, 5])
    df = pd.DataFrame(arr[None], columns=["a", "b", "c"])

    result = single_ILR_transform(df, ["a"], ["b"])
    assert result[0] == pytest.approx(1.18, abs=1e-2)

    result = single_ILR_transform(df, ["a", "b"], ["c"])
    assert result[0] == pytest.approx(1.58, abs=1e-2)


def test_single_ILR_transform():
    """Test the core functionality of a single ILR transform."""
    arr = np.array([[80, 15, 5], [75, 18, 7]])
    df = pd.DataFrame(arr, columns=["a", "b", "c"])

    result = single_ILR_transform(df, ["a"], ["b"])
    assert result[1] == pytest.approx(1.01, abs=1e-2)

    result = single_ILR_transform(df, ["a", "b"], ["c"])
    assert result[1] == pytest.approx(1.35, abs=1e-2)


# TODO: handle case where 0 is not in selected columns
def test_ilr_transform_with_zeros():
    """Test that running the transformation for a dataframe containing zeros raises the correct exception."""
    with pytest.raises(NumericValueSignException):
        arr = np.array([[80, 0, 5], [75, 18, 7]])
        df = pd.DataFrame(arr, columns=["a", "b", "c"])
        single_ILR_transform(df, ["a"], ["b"])


def test_ilr_transform_with_unexpected_column_name():
    """Test that providing an invalid column name raises the correct exception."""
    with pytest.raises(InvalidColumnException):
        arr = np.array([[65, 12, 18, 5], [63, 16, 15, 6]])
        df = pd.DataFrame(arr, columns=["a", "b", "c", "d"])
        single_ILR_transform(df, ["a", "b"], ["comp3"])
