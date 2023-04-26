import numpy as np
import pytest

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.validation.plot_rate_curve import plot_rate_curve


def test_plot_rate_curve_invalid_parameter_value():
    """Tests that invalid parameter value raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        x_values = np.linspace(0, 1, 10)
        y_values = np.linspace(0, 1, 10)
        plot_rate_curve(x_values=x_values, y_values=y_values, plot_type="scatter_plot")


def test_plot_rate_curve_x_values_out_of_bound():
    """Tests that out of bounds values raise correct exception."""
    x_values = np.arange(-4, 10)
    y_values = np.linspace(0, 1, 10)
    with pytest.raises(InvalidParameterValueException):
        plot_rate_curve(x_values=x_values, y_values=y_values, plot_type="roc")


def test_plot_rate_curve_y_values_out_of_bound():
    """Tests that out of bounds values raise correct exception."""
    y_values = np.arange(-4, 10)
    x_values = np.linspace(0, 1, 10)
    with pytest.raises(InvalidParameterValueException):
        plot_rate_curve(x_values=x_values, y_values=y_values, plot_type="roc")
