import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import InvalidColumnException
from eis_toolkit.transformations.coda.plr import single_PLR_transform


def test_single_PLR_transform_with_single_composition():
    """Test a single PLR transform operation with a single composition."""
    c_arr = np.array([80, 15, 5])
    C = pd.DataFrame(c_arr[None], columns=["a", "b", "c"])

    result = single_PLR_transform(C, "a")
    assert result[0] == pytest.approx(1.82, abs=1e-2)

    result = single_PLR_transform(C, "b")
    assert result[0] == pytest.approx(0.78, abs=1e-2)


def test_single_PLR_transform_with_simple_data():
    """Test PLR transform core functionality."""
    c_arr = np.array([[80, 15, 5], [75, 18, 7]])
    C = pd.DataFrame(c_arr, columns=["a", "b", "c"])
    result = single_PLR_transform(C, "b")
    assert result[1] == pytest.approx(0.67, abs=1e-2)


def test_single_PLR_transform_with_last_column():
    """Test that selecting the last part of the composition as the input column raises the correct exception."""
    with pytest.raises(InvalidColumnException):
        c_arr = np.array([[80, 15, 5], [75, 18, 7]])
        C = pd.DataFrame(c_arr, columns=["a", "b", "c"])
        single_PLR_transform(C, "c")
