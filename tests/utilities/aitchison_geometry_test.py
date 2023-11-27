import numpy as np
import pandas as pd

from eis_toolkit.utilities.aitchison_geometry import (
    _closure,
    _normalize,
    check_in_simplex_sample_space,
    check_in_unit_simplex_sample_space,
)

UNIT_SIMPLEX_DATAFRAME = pd.DataFrame([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.2, 0.3]])

SERIES = pd.Series([1, 2, 3, 4])

SIMPLEX_DATAFRAME = pd.DataFrame([[1, 2, 3, 4], [2, 3, 2, 3]], columns=["a", "b", "c", "d"])

NON_SIMPLEX_POSITIVE_DATAFRAME = pd.DataFrame([1, 2, 3, 4], [5, 6, 7, 8])

NON_POSITIVE_DATAFRAME = pd.DataFrame([-1, 2, 3, 4], [1, 2, 3, 4])


def test_check_for_simplex_sample_space():
    """Test whether or not a dataframe belongs to a simplex sample space is correctly identified."""
    assert check_in_simplex_sample_space(SIMPLEX_DATAFRAME)
    assert not check_in_simplex_sample_space(NON_SIMPLEX_POSITIVE_DATAFRAME)
    assert not check_in_simplex_sample_space(NON_POSITIVE_DATAFRAME)
    assert check_in_simplex_sample_space(SIMPLEX_DATAFRAME, np.float64(10))
    assert not check_in_simplex_sample_space(SIMPLEX_DATAFRAME, np.float64(100))


def test_check_for_unit_simplex_sample_space():
    """Test whether or not a dataframe belongs to the unit simplex sample space (k=1) is correctly identified."""
    assert not check_in_unit_simplex_sample_space(SIMPLEX_DATAFRAME)
    assert check_in_unit_simplex_sample_space(UNIT_SIMPLEX_DATAFRAME)


def test_normalizing():
    """Test normalizing a series to a given value."""
    output = _normalize(SERIES, sum=np.float64(20))
    expected_output = pd.Series([2, 4, 6, 8], dtype=np.float64)
    assert np.sum(output) == 20
    pd.testing.assert_series_equal(output, expected_output)


def test_normalizing_to_one():
    """Test normalizing a series to 1."""
    output = _normalize(SERIES)
    expected_output = pd.Series([0.1, 0.2, 0.3, 0.4])
    assert np.sum(output) == 1
    pd.testing.assert_series_equal(output, expected_output)


def test_closure():
    """Test the closure operation."""
    output = _closure(SIMPLEX_DATAFRAME)
    expected_output = pd.DataFrame([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.2, 0.3]], columns=["a", "b", "c", "d"])
    pd.testing.assert_frame_equal(output, expected_output)


def test_closure_of_specified_columns():
    """Test that the closure operation works with a selection of columns."""
    output = _closure(SIMPLEX_DATAFRAME, ["a", "c"])
    expected_output = pd.DataFrame([[0.25, 2, 0.75, 4], [0.5, 3, 0.5, 3]], columns=["a", "b", "c", "d"])
    pd.testing.assert_frame_equal(output, expected_output)
