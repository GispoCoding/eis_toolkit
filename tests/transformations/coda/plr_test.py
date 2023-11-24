import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import InvalidColumnException, NumericValueSignException
from eis_toolkit.transformations.coda.plr import _single_plr_transform_index, plr_transform, single_plr_transform


def test_single_plr_transform_with_single_composition():
    """Test a single PLR transform operation with a single composition."""
    arr = np.array([80, 15, 5])
    df = pd.DataFrame(arr[None], columns=["a", "b", "c"])

    result = single_plr_transform(df, "a")
    assert result[0] == pytest.approx(1.82, abs=1e-2)

    result = _single_plr_transform_index(df, 0)
    assert result[0] == pytest.approx(1.82, abs=1e-2)

    result = single_plr_transform(df, "b")
    assert result[0] == pytest.approx(0.78, abs=1e-2)

    result = _single_plr_transform_index(df, 1)
    assert result[0] == pytest.approx(0.78, abs=1e-2)


def test_single_plr_transform_with_simple_data():
    """Test the core functionality of a single PLR transform."""
    arr = np.array([[80, 15, 5], [75, 18, 7]])
    df = pd.DataFrame(arr, columns=["a", "b", "c"])
    result = single_plr_transform(df, "b")
    assert result[1] == pytest.approx(0.67, abs=1e-2)


def test_single_plr_transform_with_last_column():
    """Test that selecting the last part of the composition as the input column raises the correct exception."""
    with pytest.raises(InvalidColumnException):
        arr = np.array([[80, 15, 5], [75, 18, 7]])
        df = pd.DataFrame(arr, columns=["a", "b", "c"])
        single_plr_transform(df, "c")


def test_plr_transform():
    """Test PLR transform core functionality."""
    arr = np.array([[65, 12, 18, 5], [63, 16, 15, 6]])
    df = pd.DataFrame(arr, columns=["a", "b", "c", "d"])
    result = plr_transform(df)
    assert len(result.columns) == len(df.columns) - 1
    expected = pd.DataFrame(np.array([[1.60, 0.19, 0.91], [1.49, 0.43, 0.65]]), columns=["a", "b", "c"])
    pd.testing.assert_frame_equal(result, expected, atol=1e-2)


def test_plr_transform_with_zeros():
    """Test that running the transformation for a dataframe containing zeros raises the correct exception."""
    with pytest.raises(NumericValueSignException):
        df = pd.DataFrame(np.zeros((3, 3)), columns=["a", "b", "c"])
        plr_transform(df)
