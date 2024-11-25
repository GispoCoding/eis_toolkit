import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import InvalidColumnException
from eis_toolkit.transformations.coda.plr import _single_plr_transform_by_index, plr_transform, single_plr_transform


def test_single_plr_transform_with_single_composition():
    """Test a single PLR transform operation with a single composition."""
    arr1 = np.array([80, 15, 5])
    df1 = pd.DataFrame(arr1[None], columns=["a", "b", "c"])
    arr2 = [0, 80, 15, 5]
    df2 = pd.DataFrame([arr2], columns=["a", "b", "c", "d"])

    result = single_plr_transform(df1, numerator="a", denominator=["b", "c"])
    assert result[0] == pytest.approx(1.82, abs=1e-2)

    result = _single_plr_transform_by_index(df1, 0)
    assert result[0] == pytest.approx(1.82, abs=1e-2)

    result = single_plr_transform(df2, numerator="b")
    assert result[0] == pytest.approx(1.82, abs=1e-2)

    result = _single_plr_transform_by_index(df2, 1)
    assert result[0] == pytest.approx(1.82, abs=1e-2)


def test_single_plr_transform_with_simple_data():
    """Test the core functionality of a single PLR transform."""
    arr = np.array([[80, 15, 5], [75, 20, 5]])
    df = pd.DataFrame(arr, columns=["a", "b", "c"])
    result = single_plr_transform(df, "a", ["b", "c"])
    assert result[1] == pytest.approx(1.65, abs=1e-2)


def test_single_plr_transform_with_last_column():
    """Test that selecting the last part of the composition as the input column raises the correct exception."""
    with pytest.raises(InvalidColumnException):
        arr = np.array([[80, 15, 5], [75, 18, 7]])
        df = pd.DataFrame(arr, columns=["a", "b", "c"])
        single_plr_transform(df, "c", ["b", "c"])


def test_single_plr_invalid_inputs():
    """Test that invalid inputs raises exceptions."""
    arr = np.array([[80, 15, 5], [75, 18, 7]])
    df = pd.DataFrame(arr, columns=["a", "b", "c"])

    # Numerator not in df
    with pytest.raises(InvalidColumnException):
        single_plr_transform(df, "x", ["b", "c"])

    # Numerator in denominators
    with pytest.raises(InvalidColumnException):
        single_plr_transform(df, "a", ["a", "b"])

    # Denominators not in df
    with pytest.raises(InvalidColumnException):
        single_plr_transform(df, "a", ["x", "y"])


def test_plr_transform():
    """Test PLR transform core functionality."""
    arr = np.array([[65, 12, 18, 5], [63, 16, 15, 6]])
    df = pd.DataFrame(arr, columns=["a", "b", "c", "d"])
    result = plr_transform(df)
    assert len(result.columns) == len(df.columns) - 1
    expected = pd.DataFrame(np.array([[1.60, 0.19, 0.91], [1.49, 0.43, 0.65]]), columns=["V1", "V2", "V3"])
    pd.testing.assert_frame_equal(result, expected, atol=1e-2)


def test_plr_transform_with_columns():
    """Test PLR transform with column selection."""
    arr = np.array([[0, 65, 12, 18, 5], [0, 63, 16, 15, 6]])
    df = pd.DataFrame(arr, columns=["a", "b", "c", "d", "e"])
    result = plr_transform(df, columns=["b", "c", "d", "e"])
    assert len(result.columns) == 3
    expected = pd.DataFrame(np.array([[1.60, 0.19, 0.91], [1.49, 0.43, 0.65]]), columns=["V1", "V2", "V3"])
    pd.testing.assert_frame_equal(result, expected, atol=1e-2)
