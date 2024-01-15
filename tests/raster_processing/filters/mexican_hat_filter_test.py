from pathlib import Path

import numpy as np
import pytest
import rasterio

from eis_toolkit.exceptions import InvalidParameterValueException, InvalidRasterBandException
from eis_toolkit.raster_processing.filters.focal import mexican_hat_filter

parent_dir = Path(__file__).parent.parent
raster_path_single = parent_dir.joinpath("../data/remote/small_raster.tif")
raster_path_multi = parent_dir.joinpath("../data/remote/small_raster_multiband.tif")

idx_pixel_comparison = [15, 13]
expected_results = {
    "result_1": 7.118,
    "result_2": 3.931,
}


def test_mexican_hat_filter():
    """Test the focal filter function."""
    with rasterio.open(raster_path_single) as raster:
        result_1 = mexican_hat_filter(raster, sigma=1, truncate=1, direction="circular")
        result_2 = mexican_hat_filter(raster, sigma=2, truncate=4, size=5, direction="rectangular")

        # # Shapes and types
        assert isinstance(result_1, np.ndarray)
        assert isinstance(result_2, np.ndarray)

        assert result_1.shape == (raster.height, raster.width)
        assert result_2.shape == (raster.height, raster.width)

        # # Values
        result_1_pixel = result_1[idx_pixel_comparison[0], idx_pixel_comparison[1]]
        result_2_pixel = result_2[idx_pixel_comparison[0], idx_pixel_comparison[1]]

        np.testing.assert_almost_equal(result_1_pixel, expected_results["result_1"], decimal=3)
        np.testing.assert_almost_equal(result_2_pixel, expected_results["result_2"], decimal=3)


def test_number_bands():
    """Test if the number of bands is correct."""
    with rasterio.open(raster_path_multi) as raster:
        with pytest.raises(InvalidRasterBandException):
            mexican_hat_filter(raster)


def test_window_size():
    """Test if the window size is correct."""
    with rasterio.open(raster_path_single) as raster:
        with pytest.raises(InvalidParameterValueException):
            # Too small
            mexican_hat_filter(raster, size=0)
            mexican_hat_filter(sigma=0, truncate=0)

            # Even number
            mexican_hat_filter(raster, size=4)
