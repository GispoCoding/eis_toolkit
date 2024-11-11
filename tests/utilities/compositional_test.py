import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import InvalidCompositionException, NumericValueSignException
from eis_toolkit.transformations.coda.alr import alr_transform
from eis_toolkit.transformations.coda.clr import clr_transform
from eis_toolkit.transformations.coda.ilr import single_ilr_transform
from eis_toolkit.transformations.coda.plr import plr_transform, single_plr_transform
from eis_toolkit.utilities.checks.compositional import check_in_simplex_sample_space


def test_compositional_data_has_zeros():
    """Test that performing logratio transforms for data containing zeros raises the correct exception."""
    arr = np.array([[80, 0, 5], [75, 18, 7]])
    df = pd.DataFrame(arr, columns=["a", "b", "c"])
    with pytest.raises(NumericValueSignException):
        alr_transform(df)
    with pytest.raises(NumericValueSignException):
        clr_transform(df)
    with pytest.raises(NumericValueSignException):
        single_ilr_transform(df, ["a"], ["b"])
    with pytest.raises(NumericValueSignException):
        plr_transform(df)
    with pytest.raises(NumericValueSignException):
        single_plr_transform(df, "b")


def test_compositional_data_has_negatives():
    """Test that performing logratio transforms for data containing negative values raises the correct exception."""
    arr = np.array([[80, 25, -5], [75, 32, -7]])
    df = pd.DataFrame(arr, columns=["a", "b", "c"])
    with pytest.raises(NumericValueSignException):
        alr_transform(df)
    with pytest.raises(NumericValueSignException):
        clr_transform(df)
    with pytest.raises(NumericValueSignException):
        single_ilr_transform(df, ["a"], ["b"])
    with pytest.raises(NumericValueSignException):
        plr_transform(df)
    with pytest.raises(NumericValueSignException):
        single_plr_transform(df, "b")


def test_compositional_data_has_nans():
    """Test that performing logratio transforms for data containing NaN values raises the correct exception."""
    df = pd.DataFrame(np.ones((3, 3)), columns=["a", "b", "c"])
    df.iloc[:, 0] = np.NaN
    with pytest.raises(InvalidCompositionException):
        alr_transform(df)
    with pytest.raises(InvalidCompositionException):
        clr_transform(df)
    with pytest.raises(InvalidCompositionException):
        single_ilr_transform(df, ["a"], ["b"])
    with pytest.raises(InvalidCompositionException):
        plr_transform(df)
    with pytest.raises(InvalidCompositionException):
        single_plr_transform(df, "b")


def test_compositional_data_invalid():
    """Test that input data that does not belong to a simplex sample space raises the correct exception."""
    arr = np.array([[1, 1, 1], [2, 2, 2]])
    df = pd.DataFrame(arr, columns=["a", "b", "c"])
    with pytest.raises(InvalidCompositionException):
        alr_transform(df)
    with pytest.raises(InvalidCompositionException):
        clr_transform(df)
    with pytest.raises(InvalidCompositionException):
        single_ilr_transform(df, ["a"], ["b"])
    with pytest.raises(InvalidCompositionException):
        plr_transform(df)
    with pytest.raises(InvalidCompositionException):
        single_plr_transform(df, "b")


def test_check_for_simplex_sample_space():
    """Test whether or not a dataframe belongs to a simplex sample space is correctly identified."""
    unit_simplex_df = pd.DataFrame([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.2, 0.3]])
    simplex_df = pd.DataFrame([[1, 2, 3, 4], [2, 3, 2, 3]], columns=["a", "b", "c", "d"])
    non_simplex_positive_df = pd.DataFrame([1, 2, 3, 4], [5, 6, 7, 8])
    non_positive_df = pd.DataFrame([-1, 2, 3, 4], [1, 2, 3, 4])

    with pytest.raises(InvalidCompositionException):
        check_in_simplex_sample_space(non_simplex_positive_df)

    with pytest.raises(NumericValueSignException):
        check_in_simplex_sample_space(non_positive_df)

    with pytest.raises(InvalidCompositionException):
        check_in_simplex_sample_space(simplex_df, np.float64(100))

    # Valid cases - assert no exception is raised
    try:
        check_in_simplex_sample_space(simplex_df)
        check_in_simplex_sample_space(simplex_df, np.float64(10))
        check_in_simplex_sample_space(unit_simplex_df, np.float64(1.0))
    except Exception as ex:
        assert False, f"{type(ex)}: {ex}"
