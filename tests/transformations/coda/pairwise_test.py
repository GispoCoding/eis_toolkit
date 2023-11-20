import pytest

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.transformations.coda.pairwise import single_pairwise_logratio


def testsingle_pairwise_logratio():
    """Test the pairwise logratio transform core functionality."""
    assert single_pairwise_logratio(1.0, 1.0) == 0
    assert single_pairwise_logratio(80.0, 15.0) == pytest.approx(1.67, abs=1e-2)


def testsingle_pairwise_logratio_with_zeros():
    """Test that calling the function with a zero value as either value raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        single_pairwise_logratio(0.0, 1.0)

    with pytest.raises(InvalidParameterValueException):
        single_pairwise_logratio(1.0, 0.0)

    with pytest.raises(InvalidParameterValueException):
        single_pairwise_logratio(0.0, 0.0)
