import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import InvalidColumnException
from eis_toolkit.transformations.coda.clr import _CLR_transform

ONES_DATAFRAME_4x4 = pd.DataFrame(np.ones((4, 4)), columns=["c1", "c2", "c3", "c4"])


def test_clr_transform_contains_zeros():
    """TODO: docstring."""
    with pytest.raises(InvalidColumnException):
        df = ONES_DATAFRAME_4x4.copy()
        df.iloc[0, 0] = 0
        _CLR_transform(df)
