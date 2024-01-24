import numpy as np
import pytest
import rasterio
from beartype.typing import Sequence

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.raster_processing import reclassify_raster
from tests.raster_processing.clip_test import raster_path as SMALL_RASTER_PATH

test_array = np.array([[0, 10, 20, 30], [40, 50, 50, 60], [80, 80, 90, 90], [100, 100, 100, 100]])


def test_raster_with_defined_intervals():
    """Test raster with defined intervals."""
    interval_size = 3

    result = reclassify_raster._raster_with_defined_intervals(test_array, interval_size)

    expected_output = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])

    assert isinstance(result, Sequence)
    assert isinstance(result[0], np.ndarray)

    np.testing.assert_allclose(result[0], expected_output)


def raster_with_equal_intervals():
    """Test raster with equal intervals."""
    number_of_intervals = 10

    result = reclassify_raster._raster_with_equal_intervals(test_array, number_of_intervals)

    expected_output = np.array([[1, 2, 3, 4], [5, 6, 6, 7], [9, 9, 10, 10], [11, 11, 11, 11]])

    assert isinstance(result, Sequence)
    assert isinstance(result[0], np.ndarray)

    np.testing.assert_allclose(result[0], expected_output)


def test_raster_with_geometrical_intervals():
    """Test raster with geometrical intervals."""
    number_of_classes = 10
    nan_value = -9999

    result = reclassify_raster._raster_with_geometrical_intervals(test_array, number_of_classes, nan_value)

    assert isinstance(result, Sequence)
    assert isinstance(result[0], np.ndarray)

    expected_output = np.array([[-9, -9, -9, -9], [-9, -9, -9, -8], [8, 8, 9, 9], [9, 9, 9, 9]])

    np.testing.assert_allclose(result[0], expected_output)


def test_raster_with_manual_breaks():
    """Test raster with manual break intervals."""
    breaks = [20, 40, 60, 80]

    result = reclassify_raster._raster_with_manual_breaks(test_array, breaks)

    assert isinstance(result, Sequence)
    assert isinstance(result[0], np.ndarray)

    expected_output = np.array([[0, 0, 1, 1], [2, 2, 2, 3], [4, 4, 4, 4], [4, 4, 4, 4]])

    np.testing.assert_allclose(result[0], expected_output)


def test_raster_with_natural_breaks():
    """Test raster with natural breaks."""
    number_of_classes = 10

    result = reclassify_raster._raster_with_natural_breaks(test_array, number_of_classes)

    assert isinstance(result, Sequence)
    assert isinstance(result[0], np.ndarray)

    expected_output = np.array([[0, 1, 1, 2], [3, 4, 4, 5], [6, 6, 7, 7], [8, 8, 8, 8]])

    np.testing.assert_allclose(result[0], expected_output)


def test_raster_with_standard_deviation():
    """Test raster with standard deviation intervals."""
    number_of_intervals = 75

    result = reclassify_raster._raster_with_standard_deviation(test_array, number_of_intervals)

    assert isinstance(result, Sequence)
    assert isinstance(result[0], np.ndarray)

    expected_output = np.array([[-75, -75, -75, -36], [-25, -14, -14, -3], [20, 20, 31, 31], [75, 75, 75, 75]])

    np.testing.assert_allclose(result[0], expected_output)


def test_raster_with_quantiles():
    """Test raster with quantile intervals by."""
    number_of_quantiles = 4

    result = reclassify_raster._raster_with_quantiles(test_array, number_of_quantiles)

    assert isinstance(result, Sequence)
    assert isinstance(result[0], np.ndarray)

    expected_output = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])

    np.testing.assert_allclose(result[0], expected_output)
