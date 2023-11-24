import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import InvalidCompositionException, NumericValueSignException
from eis_toolkit.transformations.coda.alr import alr_transform
from eis_toolkit.transformations.coda.clr import clr_transform
from eis_toolkit.transformations.coda.ilr import single_ilr_transform
from eis_toolkit.transformations.coda.plr import plr_transform, single_plr_transform


def test_compositional_data_has_zeros():
    """Test that performing logratio transforms for data containing zeros raises the correct exception."""
    arr = np.array([[80, 0, 5], [75, 18, 7]])
    df = pd.DataFrame(arr, columns=["a", "b", "c"])
    with pytest.raises(NumericValueSignException):
        alr_transform(df)
        clr_transform(df)
        single_ilr_transform(df)
        plr_transform(df)
        single_plr_transform(df)


def test_compositional_data_has_negatives():
    """Test that performing logratio transforms for data containing negative values raises the correct exception."""
    arr = np.array([[80, 25, -5], [75, 32, -7]])
    df = pd.DataFrame(arr, columns=["a", "b", "c"])
    with pytest.raises(NumericValueSignException):
        alr_transform(df)
        clr_transform(df)
        single_ilr_transform(df)
        plr_transform(df)
        single_plr_transform(df)


def test_compositional_data_has_nans():
    """Test that performing logratio transforms for data containing NaN values raises the correct exception."""
    df = pd.DataFrame(np.ones((3, 3)), columns=["a", "b", "c"])
    df.iloc[:, 0] = np.NaN
    with pytest.raises(InvalidCompositionException):
        alr_transform(df)
        clr_transform(df)
        single_ilr_transform(df)
        plr_transform(df)
        single_plr_transform(df)


def test_compositional_data_invalid():
    """Test that input data that does not belong to a simplex sample space raises the correct exception."""
    arr = np.array([[1, 1, 1], [2, 2, 2]])
    df = pd.DataFrame(arr, columns=["a", "b", "c"])
    with pytest.raises(InvalidCompositionException):
        alr_transform(df)
        clr_transform(df)
        single_ilr_transform(df)
        plr_transform(df)
        single_plr_transform(df)
