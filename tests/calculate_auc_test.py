import numpy as np
import pytest

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.validation.calculate_auc import calculate_auc


def test_calculate_auc():
    """Tests that the AUC is calculated correctly."""
    assert calculate_auc(x_values=np.array([0, 1]), y_values=np.array([0, 1])) == 0.5
    assert calculate_auc(x_values=np.array([0, 1]), y_values=np.array([0, 0.5])) == 0.25
    angles = np.linspace(0, np.pi / 2, 100)
    unit_circle_first_quadrant_auc = round(calculate_auc(x_values=np.cos(angles), y_values=np.sin(angles)), 2)
    assert unit_circle_first_quadrant_auc == round(np.pi / 4, 2)


def test_calculate_auc_x_values_out_of_bound():
    """Tests that out of bounds values raise correct exception."""
    x_values = np.arange(-4, 10)
    y_values = np.linspace(0, 1, 10)
    with pytest.raises(InvalidParameterValueException):
        calculate_auc(x_values=x_values, y_values=y_values)


def test_calculate_auc_y_values_out_of_bound():
    """Tests that out of bounds values raise correct exception."""
    y_values = np.arange(-4, 10)
    x_values = np.linspace(0, 1, 10)
    with pytest.raises(InvalidParameterValueException):
        calculate_auc(x_values=x_values, y_values=y_values)
