import numpy as np
import pytest
from beartype import roar

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.prediction.fuzzy_overlay import and_overlay, gamma_overlay, or_overlay, product_overlay, sum_overlay

RASTER_DATA_1 = np.array([[1.0, 0.5, 0.2], [0.7, 0.6, 0.5], [1.0, 0.2, 0.5]])

RASTER_DATA_2 = np.array([[0.7, 0.3, 0.1], [0.4, 0.9, 0.7], [0.9, 0.2, 1.0]])

RASTERS_DATA = np.stack((RASTER_DATA_1, RASTER_DATA_2))


def test_and_overlay():
    """Test that AND overlay works as expected."""
    result = and_overlay(data=RASTERS_DATA)
    np.testing.assert_array_equal(result, np.array([[0.7, 0.3, 0.1], [0.4, 0.6, 0.5], [0.9, 0.2, 0.5]]))


def test_or_overlay():
    """Test that OR overlay works as expected."""
    result = or_overlay(data=RASTERS_DATA)
    np.testing.assert_array_equal(result, np.array([[1.0, 0.5, 0.2], [0.7, 0.9, 0.7], [1.0, 0.2, 1.0]]))


def test_product_ovelay():
    """Test that PRODUCT overlay works as expected."""
    result = product_overlay(data=RASTERS_DATA)
    np.testing.assert_allclose(result, np.array([[0.7, 0.15, 0.02], [0.28, 0.54, 0.35], [0.9, 0.04, 0.5]]))


def test_sum_overlay():
    """Test that SUM overlay works as expected."""
    result = sum_overlay(data=RASTERS_DATA)
    np.testing.assert_allclose(result, [[1.0, 0.65, 0.28], [0.82, 0.96, 0.85], [1.0, 0.36, 1.0]])


def test_gamma_overlay():
    """Test that GAMMA overlay works as expected."""
    result = gamma_overlay(data=RASTERS_DATA, gamma=0.6)
    np.testing.assert_array_equal(
        np.around(result, decimals=4), [[0.8670, 0.3616, 0.0974], [0.5335, 0.7626, 0.5960], [0.9587, 0.1495, 0.7579]]
    )


def test_gamma_overlay_equivalent_product():
    """Test that GAMMA overlay is equal to PRODUCT with 0.0 gamma."""
    result_gamma = gamma_overlay(data=RASTERS_DATA, gamma=0.0)
    result_product = product_overlay(data=RASTERS_DATA)
    np.testing.assert_array_equal(result_gamma, result_product)


def test_gamma_overlay_equivalent_sum():
    """Test that GAMMA overlay is equal to SUM with 1.0 gamma."""
    result_gamma = gamma_overlay(data=RASTERS_DATA, gamma=1.0)
    result_sum = sum_overlay(data=RASTERS_DATA)
    np.testing.assert_array_equal(result_gamma, result_sum)


def test_gamma_overlay_gamma_out_of_range():
    """Test that a gamma value out of range [0, 1] raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        gamma_overlay(data=RASTERS_DATA, gamma=3.2)


def test_overlay_data_out_of_range():
    """Test that a data value out of range [0, 1] raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        invalid_data = np.array([[1, 0.3, -0.2], [1.7, 0.6, 0.5], [1.0, 0.2, 0.5]])
        data_invalid = np.stack([RASTER_DATA_1, invalid_data])
        and_overlay(data=data_invalid)


def test_overlay_wrong_input_type():
    """Test that a wrong input parameter type raises the correct exception."""
    with pytest.raises(roar.BeartypeCallHintParamViolation):
        and_overlay(data=[1, 2, 3])
