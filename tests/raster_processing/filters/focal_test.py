from pathlib import Path

import numpy as np
import pytest
import rasterio

from eis_toolkit.exceptions import InvalidParameterValueException, InvalidRasterBandException
from eis_toolkit.raster_processing.filters.focal import focal_filter

parent_dir = Path(__file__).parent.parent
raster_path_single = parent_dir.joinpath("../data/remote/small_raster.tif")
raster_path_multi = parent_dir.joinpath("../data/remote/small_raster_multiband.tif")

idx_pixel_comparison = [15, 13]
expected_results = {
    "focal_mean": 4.326,
    "focal_median": 4.947,
}


def test_focal_filter():
    """Test the focal filter function."""
    with rasterio.open(raster_path_single) as raster:
        result_mean = focal_filter(raster, size=5, method="mean", shape="circle")
        result_median = focal_filter(raster, size=5, method="median", shape="square")

        # # Shapes and types
        assert isinstance(result_mean, np.ndarray)
        assert isinstance(result_median, np.ndarray)

        assert result_mean.shape == (raster.height, raster.width)
        assert result_median.shape == (raster.height, raster.width)

        # # Values
        result_mean_pixel = result_mean[idx_pixel_comparison[0], idx_pixel_comparison[1]]
        result_median_pixel = result_median[idx_pixel_comparison[0], idx_pixel_comparison[1]]

        np.testing.assert_almost_equal(result_mean_pixel, expected_results["focal_mean"], decimal=3)
        np.testing.assert_almost_equal(result_median_pixel, expected_results["focal_median"], decimal=3)


def test_number_bands():
    """Test if the number of bands is correct."""
    with rasterio.open(raster_path_multi) as raster:
        with pytest.raises(InvalidRasterBandException):
            focal_filter(raster)


def test_window_size():
    """Test if the window size is correct."""
    with rasterio.open(raster_path_single) as raster:
        with pytest.raises(InvalidParameterValueException):
            # Too small
            focal_filter(raster, size=0)

            # Even number
            focal_filter(raster, size=4)
