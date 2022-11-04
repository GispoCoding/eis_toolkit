import numpy as np
import pytest

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.validation.get_pa_intersection import get_pa_intersection


def test_get_pa_intersection():
    """Test that get_pa_intersection function. Two lines with slopes 1 and -1 should intercept at (0.5, 0.5)."""
    threshold_values = np.linspace(0, 1, 10)
    true_positive_rate_values = np.linspace(0, 1, 10)
    proportion_of_area_values = np.linspace(0, 1, 10)
    x, y = get_pa_intersection(
        true_positive_rate_values=true_positive_rate_values,
        proportion_of_area_values=proportion_of_area_values,
        threshold_values=threshold_values,
    )

    assert x == 0.5 and y == 0.5


def test_get_pa_intersection_true_positive_values_out_of_bounds():
    """Tests that out of bounds values raise correct exception."""
    threshold_values = np.linspace(0, 1, 10)
    proportion_of_area_values = np.linspace(0, 1, 10)
    true_positive_values = np.arange(-4, 10)
    with pytest.raises(InvalidParameterValueException):
        get_pa_intersection(
            threshold_values=threshold_values,
            true_positive_rate_values=true_positive_values,
            proportion_of_area_values=proportion_of_area_values,
        )


def test_get_pa_intersection_proportion_of_area_values_out_of_bounds():
    """Tests that out of bounds values raise correct exception."""
    threshold_values = np.linspace(0, 1, 10)
    proportion_of_area_values = np.arange(-4, 10)
    true_positive_values = np.linspace(0, 1, 10)
    with pytest.raises(InvalidParameterValueException):
        get_pa_intersection(
            threshold_values=threshold_values,
            true_positive_rate_values=true_positive_values,
            proportion_of_area_values=proportion_of_area_values,
        )
