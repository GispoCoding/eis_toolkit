from pathlib import Path

import mapclassify as mc
import numpy as np
import pytest
import rasterio

from eis_toolkit.raster_processing.reclassify_raster import (
    raster_with_defined_intervals,
    raster_with_equal_intervals,
    raster_with_geometrical_intervals,
    raster_with_manual_breaks,
    raster_with_natural_breaks,
    raster_with_quantiles,
    raster_with_standard_deviation,
)

test_dir = Path(__file__).parent.parent
raster_path = test_dir.joinpath("data/remote/small_raster.tif")
raster_copy_path = test_dir.joinpath("data/local/small_raster - Copy.tif")

band_numbers = [1]


def test_raster_with_defined_intervals():
    """Test raster with defined intervals by comparing the output of the function to numpy's digitized result."""
    with rasterio.open(raster_path) as raster:
        interval_size = 5

        output = raster_with_defined_intervals(raster, interval_size, raster_copy_path, band_numbers)

        hist, edges = np.histogram(raster.read(1), bins=interval_size)

        data = np.digitize(raster.read(1), edges)

        assert np.array_equal(data, output.read(1))


def raster_with_equal_intervals():
    """Test raster with equal intervals by comparing the output to numpy's digitized result."""
    with rasterio.open(raster_path) as raster:

        number_of_intervals = 100

        band_1 = raster.read(1)

        output = raster_with_equal_intervals(raster, number_of_intervals, raster_copy_path, band_numbers)

        expected_intervals = np.linspace(0, 100, number_of_intervals + 1)
        expected_result = np.digitize(band_1, expected_intervals)

        assert np.array_equal(output.read(1), expected_result)


def test_raster_with_geometrical_intervals():
    """Test raster with geometrical intervals by comparing the output of the function to the original data."""
    with rasterio.open(raster_path) as raster:
        number_of_classes = 10

        band_1 = raster.read(1)

        output = raster_with_geometrical_intervals(raster, number_of_classes, raster_copy_path, band_numbers)

        assert not np.array_equal(output.read(1), band_1)


def test_raster_with_manual_breaks():
    """Test raster with manual break intervals by comparing the output of the function to numpy's digitized result."""
    with rasterio.open(raster_path, "r+") as raster:

        breaks = [-2000, -1000, 500, 1000]

        band_1 = raster.read(1)

        output = raster_with_manual_breaks(raster, breaks, raster_copy_path, band_numbers)

        expected_result = np.digitize(band_1, breaks)

        assert np.array_equal(output.read(1), expected_result)


def test_raster_with_natural_breaks():
    """Test raster with natural break intervals by comparing the output of the function to MapClassify Jenks Caspall and numpy's digitized result"""
    with rasterio.open(raster_path, "r+") as raster:
        number_of_classes = 10

        band_1 = raster.read(1)

        output = raster_with_natural_breaks(raster, number_of_classes, raster_copy_path, band_numbers)

        breaks = mc.JenksCaspall(band_1, number_of_classes)
        data = np.digitize(band_1, np.sort(breaks.bins))

        assert np.array_equal(output.read(1), data)


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
    quantiles = 4

    raster = rasterio.open(raster_path, "r+")
    band_1 = raster.read(1)

    output = raster_with_quantiles(raster, quantiles, raster_copy_path, band_numbers)

    assert not np.array_equal(output.read(1), band_1)
