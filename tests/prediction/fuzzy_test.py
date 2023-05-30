import numpy as np
import pytest
from beartype import roar

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.prediction.fuzzy_overlay import FuzzyMethod, fuzzy_overlay

RASTER_DATA_1 = np.array([[1, 0.5, 0.2], [0.7, 0.6, 0.5], [1.0, 0.2, 0.5]])

RASTER_DATA_2 = np.array([[0.7, 0.3, 0.1], [0.4, 0.9, 0.7], [0.9, 0.2, 1.0]])

RASTERS_DATA = np.stack((RASTER_DATA_1, RASTER_DATA_2))


def test_fuzzy_overlay_and():
    """Test that AND overlay works as expected."""
    result = fuzzy_overlay(rasters_data=RASTERS_DATA, method=FuzzyMethod.AND)
    assert np.array_equal(result, np.array([[0.7, 0.3, 0.1], [0.4, 0.6, 0.5], [0.9, 0.2, 0.5]]))


def test_fuzzy_overlay_or():
    """Test that OR overlay works as expected."""
    result = fuzzy_overlay(rasters_data=RASTERS_DATA, method=FuzzyMethod.OR)
    assert np.array_equal(result, np.array([[1.0, 0.5, 0.2], [0.7, 0.9, 0.7], [1.0, 0.2, 1.0]]))


def test_fuzzy_overlay_product():
    """Test that PRODUCT overlay works as expected."""
    result = fuzzy_overlay(rasters_data=RASTERS_DATA, method=FuzzyMethod.PRODUCT)
    assert np.allclose(result, np.array([[0.7, 0.15, 0.02], [0.28, 0.54, 0.35], [0.9, 0.04, 0.5]]))


def test_fuzzy_overlay_sum():
    """Test that SUM overlay works as expected."""
    result = fuzzy_overlay(rasters_data=RASTERS_DATA, method=FuzzyMethod.SUM)
    assert np.allclose(result, [[1.0, 0.65, 0.28], [0.82, 0.96, 0.85], [1.0, 0.36, 1.0]])


def test_fuzzy_overlay_gamma():
    """Test that GAMMA overlay works as expected."""
    result = fuzzy_overlay(rasters_data=RASTERS_DATA, method=FuzzyMethod.GAMMA, gamma=0.6)
    assert np.array_equal(
        np.around(result, decimals=4), [[0.8670, 0.3616, 0.0974], [0.5335, 0.7626, 0.5960], [0.9587, 0.1495, 0.7579]]
    )


def test_fuzzy_overlay_missing_gamma():
    """Test that a missing gamma value when GAMMA method is selected raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        fuzzy_overlay(rasters_data=RASTERS_DATA, method=FuzzyMethod.GAMMA)


def test_fuzzy_overlay_gamma_out_of_range():
    """Test that a gamma value out of range [0, 1] raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        fuzzy_overlay(rasters_data=RASTERS_DATA, method=FuzzyMethod.GAMMA, gamma=3.2)


def test_fuzzy_overlay_data_out_of_range():
    """Test that a data value out of range [0, 1] raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        invalid_data = np.array([[1, 0.3, -0.2], [1.7, 0.6, 0.5], [1.0, 0.2, 0.5]])
        rasters_data_invalid = np.stack([RASTER_DATA_1, invalid_data])
        fuzzy_overlay(rasters_data=rasters_data_invalid, method=FuzzyMethod.AND)


def test_fuzzy_overlaye_wrong_input_type():
    """Test that a wrong input parameter type raises the correct exception."""
    with pytest.raises(roar.BeartypeCallHintParamViolation):
        fuzzy_overlay(rasters_data=[1, 2, 3], method=FuzzyMethod.AND)
