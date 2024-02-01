import numpy as np
import rasterio
from beartype.typing import Sequence

from eis_toolkit.raster_processing import reclassify_raster
from tests.raster_processing.clip_test import raster_path as SMALL_RASTER_PATH

TEST_ARRAY = np.array([[0, 10, 20, 30], [40, 50, 50, 60], [80, 80, 90, 90], [100, 100, 100, 100]])


def test_raster_with_defined_intervals():
    """Test raster with defined intervals."""
    interval_size = 3

    result = reclassify_raster._raster_with_defined_intervals(TEST_ARRAY, interval_size)

    expected_output = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])

    assert isinstance(result, Sequence)
    assert isinstance(result[0], np.ndarray)

    np.testing.assert_allclose(result[0], expected_output)


def test_raster_with_defined_intervals_main():
    """Test raster with defined intervals parameters."""
    with rasterio.open(SMALL_RASTER_PATH) as raster:
        result = reclassify_raster.raster_with_defined_intervals(
            raster=raster,
            interval_size=3,
            bands=[1],
        )
    assert isinstance(result, Sequence)
    assert isinstance(result[0], np.ndarray)


def test_raster_with_equal_intervals():
    """Test raster with equal intervals."""
    number_of_intervals = 10

    result = reclassify_raster._raster_with_equal_intervals(TEST_ARRAY, number_of_intervals)

    expected_output = np.array([[1, 2, 3, 4], [5, 6, 6, 7], [9, 9, 10, 10], [11, 11, 11, 11]])

    assert isinstance(result, Sequence)
    assert isinstance(result[0], np.ndarray)

    np.testing.assert_allclose(result[0], expected_output)


def test_raster_with_equal_intervals_main():
    """Test raster with equal intervals parameters."""
    with rasterio.open(SMALL_RASTER_PATH) as raster:
        result = reclassify_raster.raster_with_defined_intervals(
            raster=raster,
            interval_size=10,
            bands=[1],
        )
    assert isinstance(result, Sequence)
    assert isinstance(result[0], np.ndarray)


def test_raster_with_geometrical_intervals():
    """Test raster with geometrical intervals."""
    number_of_classes = 10
    nan_value = -9999

    result = reclassify_raster._raster_with_geometrical_intervals(TEST_ARRAY, number_of_classes, nan_value)

    assert isinstance(result, Sequence)
    assert isinstance(result[0], np.ndarray)

    expected_output = np.array([[-9, -9, -9, -9], [-9, -9, -9, -8], [8, 8, 9, 9], [9, 9, 9, 9]])

    np.testing.assert_allclose(result[0], expected_output)


def test_raster_with_geometrical_intervals_main():
    """Test raster with geometrical intervals parameters."""
    with rasterio.open(SMALL_RASTER_PATH) as raster:
        result = reclassify_raster.raster_with_geometrical_intervals(
            raster=raster,
            number_of_classes=10,
            nan_value=-1.0e32,
            bands=[1],
        )
    assert isinstance(result, Sequence)
    assert isinstance(result[0], np.ndarray)


def test_raster_with_manual_breaks():
    """Test raster with manual break intervals."""
    breaks = [20, 40, 60, 80]

    result = reclassify_raster._raster_with_manual_breaks(TEST_ARRAY, breaks)

    assert isinstance(result, Sequence)
    assert isinstance(result[0], np.ndarray)

    expected_output = np.array([[0, 0, 1, 1], [2, 2, 2, 3], [4, 4, 4, 4], [4, 4, 4, 4]])

    np.testing.assert_allclose(result[0], expected_output)


def test_raster_with_manual_breaks_main():
    """Test raster with manual break intervals parameters."""
    with rasterio.open(SMALL_RASTER_PATH) as raster:
        print(raster.statistics(1))
        result = reclassify_raster.raster_with_manual_breaks(
            raster=raster,
            breaks=[2, 5, 9],
            bands=[1],
        )
    assert isinstance(result, Sequence)
    assert isinstance(result[0], np.ndarray)


def test_raster_with_natural_breaks():
    """Test raster with natural breaks."""
    number_of_classes = 10

    result = reclassify_raster._raster_with_natural_breaks(TEST_ARRAY, number_of_classes)

    assert isinstance(result, Sequence)
    assert isinstance(result[0], np.ndarray)

    expected_output = np.array([[0, 1, 1, 2], [3, 4, 4, 5], [6, 6, 7, 7], [8, 8, 8, 8]])

    np.testing.assert_allclose(result[0], expected_output)


def test_raster_with_natural_breaks_main():
    """Test raster with natural break intervals parameters."""
    with rasterio.open(SMALL_RASTER_PATH) as raster:
        result = reclassify_raster.raster_with_natural_breaks(
            raster=raster,
            number_of_classes=10,
            bands=[1],
        )
    assert isinstance(result, Sequence)
    assert isinstance(result[0], np.ndarray)


def test_raster_with_standard_deviation():
    """Test raster with standard deviation intervals."""
    number_of_intervals = 75

    result = reclassify_raster._raster_with_standard_deviation(TEST_ARRAY, number_of_intervals)

    assert isinstance(result, Sequence)
    assert isinstance(result[0], np.ndarray)

    expected_output = np.array([[-75, -75, -75, -36], [-25, -14, -14, -3], [20, 20, 31, 31], [75, 75, 75, 75]])

    np.testing.assert_allclose(result[0], expected_output)


def test_raster_with_standard_deviation_main():
    """Test raster with standard_deviation intervals parameters."""
    with rasterio.open(SMALL_RASTER_PATH) as raster:
        result = reclassify_raster.raster_with_standard_deviation(
            raster=raster,
            number_of_intervals=75,
            bands=[1],
        )
    assert isinstance(result, Sequence)
    assert isinstance(result[0], np.ndarray)


def test_raster_with_quantiles():
    """Test raster with quantile intervals by."""
    number_of_quantiles = 4

    result = reclassify_raster._raster_with_quantiles(TEST_ARRAY, number_of_quantiles)

    assert isinstance(result, Sequence)
    assert isinstance(result[0], np.ndarray)

    expected_output = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])

    np.testing.assert_allclose(result[0], expected_output)


def test_raster_with_quantiles_main():
    """Test raster with quantiles parameters."""
    with rasterio.open(SMALL_RASTER_PATH) as raster:
        result = reclassify_raster.raster_with_quantiles(
            raster=raster,
            number_of_quantiles=4,
            bands=[1],
        )
    assert isinstance(result, Sequence)
    assert isinstance(result[0], np.ndarray)
