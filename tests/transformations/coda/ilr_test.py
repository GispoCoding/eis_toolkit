import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import InvalidColumnException, InvalidCompositionException, InvalidParameterValueException
from eis_toolkit.transformations.coda.ilr import _calculate_ilr_scaling_factor, single_ilr_transform


def test_calculate_scaling_factor():
    """Test the scaling factor calculation."""
    result = _calculate_ilr_scaling_factor(2, 1)
    expected = np.sqrt(2 / 3.0)  # 0.816496580927726
    assert result == expected


def test_single_ilr_transform_with_single_composition():
    """Test the core functionality of a single ILR transform with a single row of data."""
    arr = np.array([80, 15, 5]).astype(np.float64)
    df = pd.DataFrame(arr[None], columns=["a", "b", "c"])

    result = single_ilr_transform(df, ["a"], ["b"], scale=100)
    assert result[0] == pytest.approx(1.18, abs=1e-2)

    result = single_ilr_transform(df, ["a", "b"], ["c"])
    assert result[0] == pytest.approx(1.58, abs=1e-2)


def test_single_ilr_transform():
    """Test the core functionality of a single ILR transform."""
    arr = np.array([[80, 15, 5], [75, 18, 7]]).astype(dtype=np.float64)
    df = pd.DataFrame(arr, columns=["a", "b", "c"])

    result = single_ilr_transform(df, ["a"], ["b"], scale=100)
    assert result[1] == pytest.approx(1.01, abs=1e-2)

    result = single_ilr_transform(df, ["a", "b"], ["c"])
    assert result[1] == pytest.approx(1.35, abs=1e-2)


def test_ilr_transform_with_unexpected_column_name():
    """Test that providing an invalid column name raises the correct exception."""
    with pytest.raises(InvalidColumnException):
        arr = np.array([[65, 12, 18, 5], [63, 16, 15, 6]])
        df = pd.DataFrame(arr, columns=["a", "b", "c", "d"])
        single_ilr_transform(df, ["a", "b"], ["comp3"])


def test_ilr_transform_with_overlapping_subcompositions():
    """Test that providing overlapping subcomposition columns raises the correct exception."""
    with pytest.raises(InvalidCompositionException):
        arr = np.array([[65, 12, 18, 5], [63, 16, 15, 6]])
        df = pd.DataFrame(arr, columns=["a", "b", "c", "d"])
        single_ilr_transform(df, ["a", "b"], ["b"])


def test_ilr_transform_with_empty_subcomposition():
    """Test that providing an empty subcomposition list raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        arr = np.array([[65, 12, 18, 5], [63, 16, 15, 6]])
        df = pd.DataFrame(arr, columns=["a", "b", "c", "d"])
        single_ilr_transform(df, ["a", "b"], [])
