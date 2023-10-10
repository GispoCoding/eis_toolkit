from pathlib import Path
import shutil

import numpy as np
import pytest
import rasterio

from eis_toolkit.raster_processing.reclassify_raster import raster_with_defined_intervals
from eis_toolkit.raster_processing.reclassify_raster import raster_with_equal_intervals
from eis_toolkit.raster_processing.reclassify_raster import raster_with_geometrical_intervals
from eis_toolkit.raster_processing.reclassify_raster import raster_with_manual_breaks
from eis_toolkit.raster_processing.reclassify_raster import raster_with_natural_breaks
from eis_toolkit.raster_processing.reclassify_raster import raster_with_quantiles
from eis_toolkit.raster_processing.reclassify_raster import raster_with_standard_deviation

parent_dir = Path(__file__).parent

source = "data/remote/small_raster.tif"
target = "data/remote/small_raster - Copy.tif"
shutil.copy(parent_dir.joinpath(source), parent_dir.joinpath(target))
raster_path = parent_dir.joinpath(target)

band_numbers = [1]

def test_raster_with_defined_intervals():
    """Test raster with defined intervals by comparing the output of the function to the original data."""
    interval_size = 5
    
    raster = rasterio.open(raster_path, 'r+')
    band_1 = raster.read(1)

    output = raster_with_defined_intervals(raster, interval_size, band_numbers)

    assert not np.array_equal(output.read(1), band_1)

def raster_with_equal_intervals():
    """Test raster with equal intervals by comparing the output of the function to the original data."""
    number_of_intervals = 100

    raster = rasterio.open(raster_path, 'r+')
    band_1 = raster.read(1)

    output = raster_with_equal_intervals(raster, number_of_intervals, band_numbers)

    assert not np.array_equal(output.read(1), band_1)


def test_raster_with_geometrical_intervals():
    """Test raster with geometrical intervals by comparing the output of the function to the original data."""
    number_of_classes = 10

    raster = rasterio.open(raster_path, 'r+')
    band_1 = raster.read(1)

    output = raster_with_geometrical_intervals(raster, number_of_classes, band_numbers)

    assert not np.array_equal(output.read(1), band_1)


def test_raster_with_manual_breaks():
    """Test raster with manual break intervals by comparing the output of the function to the original data."""
    breaks = [-2000, -1000, 500, 1000]

    raster = rasterio.open(raster_path, 'r+')
    band_1 = raster.read(1)

    output = raster_with_manual_breaks(raster, breaks, band_numbers)

    assert not np.array_equal(output.read(1), band_1)
    
    
def test_raster_with_natural_breaks():
    """Test raster with natural break intervals by comparing the output of the function to the original data."""
    number_of_classes = 10

    raster = rasterio.open(raster_path, 'r+')
    band_1 = raster.read(1)

    output = raster_with_natural_breaks(raster, number_of_classes, band_numbers)

    assert not np.array_equal(output.read(1), band_1)


def raster_with_standard_deviation():
    """Test raster with standard deviation intervals by comparing the output of the function to the original data."""
    intervals = 75

    raster = rasterio.open(raster_path, 'r+')
    band_1 = raster.read(1)

    output = raster_with_standard_deviation(raster, intervals, band_numbers)

    assert not np.array_equal(output.read(1), band_1)


def test_raster_with_quantiles():
    """Test raster with quantile intervals by comparing the output of the function to the original data."""
    quantiles = 4

    raster = rasterio.open(raster_path, 'r+')
    band_1 = raster.read(1)

    output = raster_with_quantiles(raster, quantiles, band_numbers)

    assert not np.array_equal(output.read(1), band_1)
    