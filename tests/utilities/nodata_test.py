from pathlib import Path

import numpy as np
import rasterio

from eis_toolkit.utilities.nodata import (
    handle_nodata_as_nan,
    nan_to_nodata,
    nodata_to_nan,
    replace_raster_nodata_each_band,
    set_nodata_raster_meta,
)

parent_dir = Path(__file__).parent.parent
raster_path = parent_dir.joinpath("data/remote/small_raster.tif")
multi_raster_path = parent_dir.joinpath("data/remote/small_raster_multiband.tif")


def test_set_nodata_raster_meta():
    """Test that setting raster nodata works as intended."""
    raster = rasterio.open(raster_path)
    new_meta = set_nodata_raster_meta(raster.meta, 10)
    assert new_meta["nodata"] == 10


def test_replace_raster_nodata_each_band():
    """Test that replacing replacing nodata for each band separately works as expected."""
    multiband_data = np.array([[[1, 2, 3, 2, 1], [2, 3, 5, 4, 3]], [[15, 16, 16, 17, 12], [16, 13, 10, 14, 16]]])
    nodata_per_band = {1: 2, 2: [16, 17]}
    target_data = np.array(
        [[[1, -9999, 3, -9999, 1], [-9999, 3, 5, 4, 3]], [[15, -9999, -9999, -9999, 12], [-9999, 13, 10, 14, -9999]]]
    )
    bands_replaced = replace_raster_nodata_each_band(multiband_data, nodata_per_band)
    assert np.array_equal(bands_replaced, target_data)


def test_nodata_to_nan():
    """Test that replacing specified nodata with np.nan works as expected."""
    data = np.array([[1, 2, 3, 2, 1], [2, 3, 5, 4, 3]])
    target_data = np.array([[np.nan, 2, 3, 2, np.nan], [2, 3, 5, 4, 3]])
    converted_data = nodata_to_nan(data, nodata_value=1)
    assert np.allclose(converted_data, target_data, equal_nan=True)


def test_nan_to_nodata():
    """Test that replacing np.nan with specified nodata works as expected."""
    data = np.array([[1, np.nan, 3, np.nan, 1], [np.nan, 3, 5, 4, 3]])
    target_data = np.array([[1, 2, 3, 2, 1], [2, 3, 5, 4, 3]])
    converted_data = nan_to_nodata(data, nodata_value=2)
    assert np.array_equal(converted_data, target_data)


def test_handle_nodata_as_nan():
    """Test that the nodata handling decorator works as expected."""

    @handle_nodata_as_nan
    def dummy_func(data, dummy_variable):
        dummy_variable = dummy_variable
        return data

    data = np.array([[1, 2, 3, 2, 1], [2, 3, 5, 4, 3]])
    processed_data = dummy_func(data, dummy_variable=5, nodata_value=2)
    assert np.array_equal(processed_data, data)
