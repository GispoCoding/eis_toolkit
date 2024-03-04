import numpy as np
import pandas as pd
import pytest

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.exploratory_analyses.chi_square_test import chi_square_test

DATA = pd.DataFrame({"e": [0, 0, 1, 1], "f": [True, False, True, True]})


def test_chi_square_test():
    """Test that returned statistics for independence are correct."""
    output_statistics = chi_square_test(data=DATA, target_column="e", columns=["f"])
    np.testing.assert_array_equal(list(output_statistics["f"].values()), [0.0, 1.0, 1])


def test_invalid_target_column():
    """Test that invalid target column raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        chi_square_test(data=DATA, target_column="invalid_column")
