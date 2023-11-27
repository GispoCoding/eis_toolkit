import numpy as np
import pandas as pd

from eis_toolkit.utilities.aitchison_geometry import _closure, _normalize


def test_normalizing():
    """Test normalizing a series to a given value."""
    output = _normalize(pd.Series([1, 2, 3, 4]), sum=np.float64(20))
    expected_output = pd.Series([2, 4, 6, 8], dtype=np.float64)
    assert np.sum(output) == 20
    pd.testing.assert_series_equal(output, expected_output)


def test_normalizing_to_one():
    """Test normalizing a series to 1."""
    output = _normalize(pd.Series([1, 2, 3, 4]))
    expected_output = pd.Series([0.1, 0.2, 0.3, 0.4])
    assert np.sum(output) == 1
    pd.testing.assert_series_equal(output, expected_output)


def test_closure():
    """Test the closure operation."""
    df = pd.DataFrame([[1, 2, 3, 4], [2, 3, 2, 3]], columns=["a", "b", "c", "d"], dtype=np.float64)
    output = _closure(df)
    expected_output = pd.DataFrame([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.2, 0.3]], columns=["a", "b", "c", "d"])
    pd.testing.assert_frame_equal(output, expected_output)
