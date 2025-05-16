from pathlib import Path

import numpy as np
import rasterio

from eis_toolkit.utilities.nodata import (
    convert_raster_nodata,
    handle_nodata_as_nan,
    nan_to_nodata,
    nodata_to_nan,
    replace_with_nodata,
    set_raster_nodata,
    unify_raster_nodata,
)
from tests.raster_processing.clip_test import raster_path as SMALL_RASTER_PATH

test_dir = Path(__file__).parent.parent
multi_raster_path = test_dir.joinpath("data/remote/small_raster_multiband.tif")


def test_set_nodata_raster_meta():
    """Test that setting raster nodata works as intended."""
    with rasterio.open(SMALL_RASTER_PATH) as raster:
        new_meta = set_raster_nodata(raster.meta, 10)
        assert new_meta["nodata"] == 10


def test_convert_raster_nodata():
    """Test that converting raster nodata works as expected."""
    with rasterio.open(SMALL_RASTER_PATH) as raster:
        raster_data = raster.read()
        # Picking a random value that is known to be present in raster since the dataset does not have nodata cells
        old_nodata = 8.8060
        new_nodata = -999
        assert np.count_nonzero(raster_data == old_nodata) > 0
        assert np.count_nonzero(raster_data == new_nodata) == 0
        out_image, out_meta = convert_raster_nodata(raster, old_nodata=8.8060, new_nodata=-999)
        assert np.count_nonzero(out_image == new_nodata) > 0
        assert out_meta["nodata"] == -999


def test_replace_with_nodata():
    """Test that replacing raster pixel values with nodata works as expected."""
    target_value = 2.705
    nodata_value = -999

    with rasterio.open(SMALL_RASTER_PATH) as raster:
        raster_data = raster.read()
        nr_of_pixels = np.count_nonzero(raster_data == target_value)
        assert nr_of_pixels > 0
        assert np.count_nonzero(raster_data == nodata_value) == 0

        replace_condition = "equal"
        out_image, out_meta = replace_with_nodata(raster, target_value, nodata_value, replace_condition)
        assert np.count_nonzero(out_image == nodata_value) > 0
        assert np.count_nonzero(out_image == target_value) == 0
        assert out_meta["nodata"] == -999

    with rasterio.open(SMALL_RASTER_PATH) as raster:
        raster_data = raster.read()
        # Ensure some pixels exist that are less than the target value
        nr_of_pixels_less_than_target = np.count_nonzero(raster_data < target_value)
        assert nr_of_pixels_less_than_target > 0

        replace_condition = "less_than"
        out_image, out_meta = replace_with_nodata(raster, target_value, nodata_value, replace_condition)

        assert np.count_nonzero(out_image == nodata_value) == nr_of_pixels_less_than_target
        assert np.count_nonzero((out_image < target_value) & (out_image != nodata_value)) == 0
        assert out_meta["nodata"] == -999

    with rasterio.open(SMALL_RASTER_PATH) as raster:
        raster_data = raster.read()
        # Ensure some pixels exist that are greater than the target value
        nr_of_pixels_greater_than_target = np.count_nonzero(raster_data > target_value)
        assert nr_of_pixels_greater_than_target > 0

        replace_condition = "greater_than"
        out_image, out_meta = replace_with_nodata(raster, target_value, nodata_value, replace_condition)

        assert np.count_nonzero(out_image == nodata_value) == nr_of_pixels_greater_than_target
        assert np.count_nonzero(out_image > target_value) == 0
        assert out_meta["nodata"] == -999

    with rasterio.open(SMALL_RASTER_PATH) as raster:
        raster_data = raster.read()
        # Ensure some pixels exist that are less than or equal to the target value
        nr_of_pixels_less_than_or_equal = np.count_nonzero(raster_data <= target_value)
        assert nr_of_pixels_less_than_or_equal > 0

        replace_condition = "less_than_or_equal"
        out_image, out_meta = replace_with_nodata(raster, target_value, nodata_value, replace_condition)

        assert np.count_nonzero(out_image == nodata_value) == nr_of_pixels_less_than_or_equal
        assert np.count_nonzero((out_image <= target_value) & (out_image != nodata_value)) == 0
        assert out_meta["nodata"] == -999

    with rasterio.open(SMALL_RASTER_PATH) as raster:
        raster_data = raster.read()
        # Ensure some pixels exist that are greater than or equal to the target value
        nr_of_pixels_greater_than_or_equal = np.count_nonzero(raster_data >= target_value)
        assert nr_of_pixels_greater_than_or_equal > 0

        replace_condition = "greater_than_or_equal"
        out_image, out_meta = replace_with_nodata(raster, target_value, nodata_value, replace_condition)

        assert np.count_nonzero(out_image == nodata_value) == nr_of_pixels_greater_than_or_equal
        assert np.count_nonzero(out_image >= target_value) == 0
        assert out_meta["nodata"] == -999


def test_unify_raster_nodata():
    """Test that unifying raster nodata for multiple rasters works as expected."""
    with rasterio.open(SMALL_RASTER_PATH) as raster:
        with rasterio.open(SMALL_RASTER_PATH) as raster_2:
            raster_out_1, raster_out_2 = unify_raster_nodata([raster, raster_2], new_nodata=-999)
            assert raster_out_1[1]["nodata"] == -999
            assert raster_out_2[1]["nodata"] == -999


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
