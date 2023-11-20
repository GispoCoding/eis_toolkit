from pathlib import Path

import mapclassify as mc
import numpy as np
import pytest
import rasterio

from eis_toolkit.raster_processing.reclassify_raster import raster_with_standard_deviation

test_dir = Path(__file__).parent.parent
raster_path = test_dir.joinpath("data/remote/small_raster.tif")

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
    number_of_classes = 10

    median_value = np.nanmedian(test_array)
    max_value = np.nanmax(test_array)
    min_value = np.nanmin(test_array)
    print(median_value)

    test_array_flat = test_array.flatten()
    test_array_flat = np.ma.masked_where(np.isnan(test_array_flat), test_array_flat)

    values_out = np.zeros_like(test_array_flat)  # The same shape as the flattened array

    if (median_value - min_value) < (max_value - median_value):  # Large end tail longer
        raster_half = test_array_flat[np.where((test_array_flat > median_value) & (test_array_flat != np.nan))]
        range_half = max_value - median_value
        raster_half = raster_half - median_value + (range_half) / 1000.0
    else:  # Small end tail longer
        raster_half = test_array_flat[np.where((test_array_flat < median_value) & (test_array_flat != np.nan))]
        range_half = median_value - min_value
        raster_half = raster_half - min_value + (range_half) / 1000.0

    min_half = np.nanmin(raster_half)
    max_half = np.nanmax(raster_half)

    # Number of classes
    fac = (max_half / min_half) ** (1 / number_of_classes)

    ibp = 1
    brpt_half = [min_half]
    brpt = [min_half]
    width = [0]

    while brpt[-1] < max_half:
        ibp += 1
        brpt.append(min_half * fac ** (ibp - 1))
        brpt_half.append(brpt[-1])
        width.append(brpt_half[-1] - brpt_half[0])

    k = 0

    for j in range(1, len(width) - 2):
        values_out[
            np.where(
                ((median_value + width[j]) < test_array_flat)
                & (test_array_flat <= (median_value + width[j + 1]))
                & (test_array_flat != np.nan)
            )
        ] = (j + 1)
        values_out[
            np.where(
                ((median_value - width[j]) > test_array_flat)
                & (test_array_flat >= (median_value - width[j + 1]))
                & (test_array_flat != np.nan)
            )
        ] = (-j - 1)
        k = j

    values_out[np.where(((median_value + width[k + 1]) < test_array_flat) & (test_array_flat != np.nan))] = k + 1
    values_out[np.where(((median_value - width[k + 1]) > test_array_flat) & (test_array_flat != np.nan))] = -k - 1
    values_out[np.where(median_value == test_array_flat)] = 0

    values_out = values_out.reshape(test_array.shape)

    expected_output = np.array([[-9, -9, -9, -9], [-9, -9, -9, -8], [8, 8, 9, 9], [9, 9, 9, 9]])

    np.testing.assert_allclose(values_out, expected_output)


def test_raster_with_manual_breaks():
    """Test raster with manual break intervals by comparing the output of the function to numpy's digitized result."""
    breaks = [20, 40, 60, 80]

    data = np.digitize(test_array, breaks)

    expected_output = np.array([[0, 0, 1, 1], [2, 2, 2, 3], [4, 4, 4, 4], [4, 4, 4, 4]])

    np.testing.assert_allclose(data, expected_output)


def test_raster_with_natural_breaks():
    """Test raster with natural break intervals by comparing the output of the function to MapClassify's Jenks Caspall and numpy's digitized result."""
    number_of_classes = 10

    breaks = mc.JenksCaspall(test_array, number_of_classes)
    data = np.digitize(test_array, np.sort(breaks.bins))

    expected_output = np.array([[0, 1, 1, 2], [3, 4, 4, 5], [6, 6, 7, 7], [8, 8, 8, 8]])

    np.testing.assert_allclose(data, expected_output)


def test_raster_with_standard_deviation():
    """Test raster with standard deviation intervals by comparing the output of the function to the original data."""
    with rasterio.open(raster_path) as raster:
        number_of_intervals = 75

        out_img, out_meta = raster_with_standard_deviation(raster, number_of_intervals, band_numbers)

        band = raster.read(1)

        statistics = raster.statistics(1)
        stddev = statistics.std
        mean = statistics.mean
        interval_size = 2 * stddev / number_of_intervals

        classified = np.empty_like(band)

        below_mean = band < (mean - stddev)
        above_mean = band > (mean + stddev)

        classified[below_mean] = -number_of_intervals
        classified[above_mean] = number_of_intervals

        in_between = ~below_mean & ~above_mean
        interval = ((band - (mean - stddev)) / interval_size).astype(int)
        classified[in_between] = interval[in_between] - number_of_intervals // 2

        np.testing.assert_allclose(out_img[0], classified)


def test_raster_with_quantiles():
    """Test raster with quantile intervals by comparing the output of the function to the original data."""
    number_of_quantiles = 4

    intervals = [np.percentile(test_array, i * 100 / number_of_quantiles) for i in range(number_of_quantiles)]
    data = np.digitize(test_array, intervals)

    expected_output = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])

    np.testing.assert_allclose(data, expected_output)
