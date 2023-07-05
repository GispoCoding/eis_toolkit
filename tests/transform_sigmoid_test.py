import pytest
import rasterio
import numpy as np
from pathlib import Path

from eis_toolkit.transformations import sigmoid
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


def test_sigmoid_transform():
    """Test that transformation works as intended"""
    bands = None
    bounds = [(0, 1)]
    slope = [1]
    center = True
    nodata = 3.748

    with rasterio.open(raster_path) as raster:
        out_array, out_meta, out_settings = sigmoid.sigmoid_transform(
            raster=raster, bands=bands, bounds=bounds, slope=slope, center=center, nodata=nodata
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
        bounds = bounds * out_meta["count"]
        slope = slope * out_meta["count"]
        out_decimals = set_max_precision()
        test_array = cast_array_to_float(test_array, cast_int=True)

        for i in range(0, out_meta["count"]):
            lower = bounds[i][0]
            upper = bounds[i][1]

            test_array[i] = replace_values(test_array[i], values_to_replace=[nodata, np.inf], replace_value=np.nan)

            if center == True:
                test_array[i] = test_array[i] - np.nanmean(test_array[i])

            test_array[i] = lower + (upper - lower) * (1 / (1 + np.exp(-slope[i] * (test_array[i]))))

        test_array = truncate_decimal_places(test_array, decimal_places=out_decimals)
        test_array = nan_to_nodata(test_array, nodata_value=nodata)
        test_array = cast_array_to_float(test_array, scalar=nodata, cast_float=True)

        assert out_array.shape == (out_meta["count"], raster.height, raster.width)
        assert out_array.dtype == test_array.dtype
        np.testing.assert_array_equal(out_array, test_array)


def test_sigmoid_band_selection():
    """Tests that invalid parameter value for band selection raises the correct exception."""
    with pytest.raises(InvalidRasterBandException):
        with rasterio.open(raster_path) as raster:
            sigmoid.sigmoid_transform(raster=raster, bands=[0], bounds=[(0, 1)], slope=[1], center=True, nodata=None)
            sigmoid.sigmoid_transform(
                raster=raster, bands=list(range(1, 100)), bounds=[(0, 1)], slope=[1], center=True, nodata=None
            )


def test_sigmoid_parameter_length():
    """Tests that invalid length for provided parameters raises the correct exception."""
    with pytest.raises(NonMatchingParameterLengthsException):
        with rasterio.open(raster_path) as raster:
            # Invalid bounds
            sigmoid.sigmoid_transform(
                raster=raster, bands=[1, 2, 3], bounds=[(0, 1), (0, 2)], slope=[1], center=True, nodata=None
            )

            # Invalid slope
            sigmoid.sigmoid_transform(
                raster=raster, bands=[1, 2, 3], bounds=[(0, 1)], slope=[1, 1], center=True, nodata=None
            )


def test_sigmoid_min_max_position():
    """Tests that invalid min-max positions for provided parameters raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        with rasterio.open(raster_path) as raster:
            # Invalid position of minimum and maximum values for new_range
            sigmoid.sigmoid_transform(raster=raster, bands=None, bounds=[(0, 0)], slope=[1], center=True, nodata=None)
            sigmoid.sigmoid_transform(raster=raster, bands=None, bounds=[(1, 0)], slope=[1], center=True, nodata=None)
