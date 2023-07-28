import numpy as np
import pytest

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.validation.plot_prediction_area_curves import plot_prediction_area_curves


def test_plot_prediction_area_curves_true_positive_values_out_of_bounds():
    """Tests that out of bounds values raise correct exception."""
    threshold_values = np.linspace(0, 1, 10)
    proportion_of_area_values = np.linspace(0, 1, 10)
    true_positive_values = np.arange(-4, 10)
    with pytest.raises(InvalidParameterValueException):
        plot_prediction_area_curves(
            threshold_values=threshold_values,
            true_positive_rate_values=true_positive_values,
            proportion_of_area_values=proportion_of_area_values,
        )


def test_plot_prediction_area_curves_proportion_of_area_values_out_of_bounds():
    """Tests that out of bounds values raise correct exception."""
    threshold_values = np.linspace(0, 1, 10)
    proportion_of_area_values = np.arange(-4, 10)
    true_positive_values = np.linspace(0, 1, 10)
    with pytest.raises(InvalidParameterValueException):
        plot_prediction_area_curves(
            threshold_values=threshold_values,
            true_positive_rate_values=true_positive_values,
            proportion_of_area_values=proportion_of_area_values,
        )
