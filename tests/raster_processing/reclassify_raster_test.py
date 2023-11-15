from pathlib import Path

import mapclassify as mc
import numpy as np
import pytest
import rasterio

from eis_toolkit.raster_processing.reclassify_raster import (
    raster_with_geometrical_intervals,
    raster_with_standard_deviation,
)

test_dir = Path(__file__).parent.parent
raster_path = test_dir.joinpath("data/remote/small_raster.tif")
raster_copy_path = test_dir.joinpath("data/local/small_raster - Copy.tif")

band_numbers = [1]

test_array = np.array([[0, 10, 20, 30], [40, 50, 50, 60], [80, 80, 90, 90], [100, 100, 100, 100]])


def test_raster_with_defined_intervals():
    """Test raster with defined intervals by comparing the output of the function to numpy's digitized result."""
    interval_size = 3

    _, edges = np.histogram(test_array, bins=interval_size)

    data = np.digitize(test_array, edges)

    expected_output = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])

    np.testing.assert_allclose(data, expected_output)


def raster_with_equal_intervals():
    """Test raster with equal intervals by comparing the output to numpy's digitized result."""
    number_of_intervals = 10

    expected_intervals = np.linspace(0, 100, number_of_intervals)
    data = np.digitize(test_array, expected_intervals)

    expected_output = np.array([[1, 2, 3, 4], [5, 6, 6, 7], [9, 9, 10, 10], [11, 11, 11, 11]])

    np.testing.assert_allclose(data, expected_output)


def test_raster_with_geometrical_intervals():
    """Test raster with geometrical intervals by comparing the output of the function to the original data."""
    with rasterio.open(raster_path) as raster:
        number_of_classes = 10

        band_1 = raster.read(1)

        output = raster_with_geometrical_intervals(raster, number_of_classes, raster_copy_path, band_numbers)

        assert not np.array_equal(output.read(1), band_1)


def test_raster_with_manual_breaks():
    """Test raster with manual break intervals by comparing the output of the function to numpy's digitized result."""
    breaks = [20, 40, 60, 80]

    data = np.digitize(test_array, breaks)

    expected_output = np.array([[0, 0, 1, 1], [2, 2, 2, 3], [4, 4, 4, 4], [4, 4, 4, 4]])

    np.testing.assert_allclose(data, expected_output)


def test_raster_with_natural_breaks():
    """Test raster with natural break intervals by comparing the output of the function
    to MapClassify's Jenks Caspall and numpy's digitized result"""
    number_of_classes = 10

    breaks = mc.JenksCaspall(test_array, number_of_classes)
    data = np.digitize(test_array, np.sort(breaks.bins))

    expected_output = np.array([[0, 1, 1, 2], [3, 4, 4, 5], [6, 6, 7, 7], [8, 8, 8, 8]])

    np.testing.assert_allclose(data, expected_output)


def raster_with_standard_deviation():
    """Test raster with standard deviation intervals by comparing the output of the function to the original data."""
    with rasterio.open(raster_path) as raster:
        number_of_intervals = 75

        output = raster_with_standard_deviation(raster, number_of_intervals, raster_copy_path, band_numbers)

        band = raster.read(1)

        stddev = raster.statistics(band + 1).std
        mean = raster.statistics(band + 1).mean
        interval_size = 2 * stddev / number_of_intervals

        classified = np.empty_like(band)

        below_mean = band < (mean - stddev)
        above_mean = band > (mean + stddev)

        classified[below_mean] = -number_of_intervals
        classified[above_mean] = number_of_intervals

        in_between = ~below_mean & ~above_mean
        interval = ((band - (mean - stddev)) / interval_size).astype(int)
        classified[in_between] = interval[in_between] - number_of_intervals // 2

        assert np.array_equal(output.read(1), classified)


def test_raster_with_quantiles():
    """Test raster with quantile intervals by comparing the output of the function to the original data."""
    number_of_quantiles = 4

    intervals = [np.percentile(test_array, i * 100 / number_of_quantiles) for i in range(number_of_quantiles)]
    data = np.digitize(test_array, intervals)

    expected_output = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])

    np.testing.assert_allclose(data, expected_output)
