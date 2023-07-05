import pytest
import rasterio
import numpy as np
from pathlib import Path

from eis_toolkit.transformations import linear
from eis_toolkit.utilities.miscellaneous import (
    replace_values,
    truncate_decimal_places,
    set_max_precision,
    cast_array_to_float,
)
from eis_toolkit.utilities.nodata import nan_to_nodata
from eis_toolkit.exceptions import (
    InvalidRasterBandException,
    NonMatchingParameterLengthsException,
    InvalidParameterValueException,
)

parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("data/remote/small_raster_multiband.tif")


def test_z_score_normalization():
    """Test that transformation works as intended"""
    bands = None
    nodata = 3.748

    with rasterio.open(raster_path) as raster:
        out_array, out_meta, out_settings = linear.z_score_normalization(raster=raster, bands=bands, nodata=nodata)

        # Output shapes and types
        assert isinstance(out_array, np.ndarray)
        assert isinstance(out_meta, dict)
        assert isinstance(out_settings, dict)

        # Output array (nodata in place)
        test_array = raster.read(list(range(1, out_meta["count"] + 1)))
        
        np.testing.assert_array_equal(
            np.ma.masked_values(out_array, value=nodata, shrink=False).mask,
            np.ma.masked_values(test_array, value=nodata, shrink=False).mask,
        )

        # Output array (transformation result)
        out_decimals = set_max_precision()
        test_array = cast_array_to_float(test_array, cast_int=True)
        test_array = replace_values(test_array, values_to_replace=[nodata, np.inf], replace_value=np.nan)
        
        for i in range(0, out_meta["count"]):
            test_array[i] = (test_array[i] - float(np.nanmean(test_array[i]))) / float(np.nanstd(test_array[i]))

        test_array = truncate_decimal_places(test_array, decimal_places=out_decimals)
        test_array = nan_to_nodata(test_array, nodata_value=nodata)
        test_array = cast_array_to_float(test_array, scalar=nodata, cast_float=True)

        assert out_array.shape == (out_meta["count"], raster.height, raster.width)
        assert out_array.dtype == test_array.dtype
        np.testing.assert_array_equal(out_array, test_array)


def test_min_max_scaling():
    """Test that transformation works as intended"""
    bands = None
    nodata = 3.748
    new_range = [(0, 1)]

    with rasterio.open(raster_path) as raster:
        out_array, out_meta, out_settings = linear.min_max_scaling(
            raster=raster, bands=bands, new_range=[(0, 1)], nodata=nodata
        )

        # Output shapes and types
        assert isinstance(out_array, np.ndarray)
        assert isinstance(out_meta, dict)
        assert isinstance(out_settings, dict)

        # Output array (nodata in place)
        test_array = raster.read(list(range(1, out_meta["count"] + 1)))

        np.testing.assert_array_equal(
            np.ma.masked_values(out_array, value=nodata, shrink=False).mask,
            np.ma.masked_values(test_array, value=nodata, shrink=False).mask,
        )

        # Output array (transformation result)
        new_range = new_range * out_meta["count"]
        out_decimals = set_max_precision()
        test_array = cast_array_to_float(test_array, cast_int=True)
        test_array = replace_values(test_array, values_to_replace=[nodata, np.inf], replace_value=np.nan)
        
        for i in range(0, out_meta["count"]):
            min = np.nanmin(test_array[i])
            max = np.nanmax(test_array[i])
            new_min = new_range[i][0]
            new_max = new_range[i][1]

            scaler = (test_array[i] - min) / (max - min)
            test_array[i] = (scaler * (new_max - new_min)) + new_min

        test_array = truncate_decimal_places(test_array, decimal_places=out_decimals)
        test_array = nan_to_nodata(test_array, nodata_value=nodata)
        test_array = cast_array_to_float(test_array, scalar=nodata, cast_float=True)

        assert out_array.shape == (out_meta["count"], raster.height, raster.width)
        assert out_array.dtype == test_array.dtype
        np.testing.assert_array_equal(out_array, test_array)


def test_linear_band_selection():
    """Tests that invalid parameter value for band selection raises the correct exception."""
    with pytest.raises(InvalidRasterBandException):
        with rasterio.open(raster_path) as raster:
            linear.z_score_normalization(raster=raster, bands=[0], nodata=None)
            linear.z_score_normalization(raster=raster, bands=list(range(1, 100)), nodata=None)
            linear.min_max_scaling(raster=raster, bands=[0], new_range=[(0, 1)], nodata=None)
            linear.min_max_scaling(raster=raster, bands=list(range(1, 100)), new_range=[(0, 1)], nodata=None)


def test_linear_parameter_length():
    """Tests that invalid length for provided parameters raises the correct exception."""
    with pytest.raises(NonMatchingParameterLengthsException):
        with rasterio.open(raster_path) as raster:
            # Invalid new_range
            linear.min_max_scaling(raster=raster, bands=[1, 2, 3], new_range=[(0, 1), (0, 2)], nodata=None)
            linear.min_max_scaling(raster=raster, bands=[1, 2], new_range=[(0, 1), (0, 2), (0, 3)], nodata=None)


def test_linear_min_max_position():
    """Tests that invalid min-max positions for provided parameters raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        with rasterio.open(raster_path) as raster:
            # Invalid position of minimum and maximum values for new_range
            linear.min_max_scaling(raster=raster, bands=None, new_range=[(1, 0)], nodata=None)
            linear.min_max_scaling(raster=raster, bands=[1, 2, 3], new_range=[(0, 0), (1, 0), (2, 0)], nodata=None)
