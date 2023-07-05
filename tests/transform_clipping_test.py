import pytest
import rasterio
import numpy as np
from pathlib import Path

from eis_toolkit.transformations import clipping
from eis_toolkit.utilities.miscellaneous import (
    cast_array_to_int,
    cast_scalar_to_int,
    cast_array_to_float,
)
from eis_toolkit.utilities.nodata import nan_to_nodata, nodata_to_nan
from eis_toolkit.exceptions import (
    InvalidRasterBandException,
    NonMatchingParameterLengthsException,
    InvalidParameterValueException,
)

parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("data/remote/small_raster_multiband.tif")


def test_clipping():
    """Test that transformation works as intended"""
    bands = None
    limits = [(-1, 1)]

    nodata = 3.748

    with rasterio.open(raster_path) as raster:
        out_array, out_meta, out_settings = clipping.clipping(raster=raster, bands=bands, limits=limits, nodata=nodata)

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
        limits = limits * out_meta["count"]
        initial_dtype = test_array.dtype
        test_array = cast_array_to_float(test_array, cast_int=True)
        test_array = nodata_to_nan(test_array, nodata_value=nodata)

        for i in range(0, out_meta["count"]):
            lower = limits[i][0]
            upper = limits[i][1]

            test_array[i] = np.where(test_array[i] < lower, lower, test_array[i])
            test_array[i] = np.where(test_array[i] > upper, upper, test_array[i])

        test_array = nan_to_nodata(test_array, nodata_value=nodata)
        test_array = cast_array_to_int(test_array, scalar=nodata, initial_dtype=initial_dtype)

        assert out_array.shape == (out_meta["count"], raster.height, raster.width)
        assert out_array.dtype == test_array.dtype
        np.testing.assert_array_equal(out_array, test_array)


def test_clipping_band_selection():
    """Tests that invalid parameter value for band selection raises the correct exception."""
    with pytest.raises(InvalidRasterBandException):
        with rasterio.open(raster_path) as raster:
            clipping.clipping(raster=raster, bands=[0], limits=[(-1, 1)], nodata=None)
            clipping.clipping(raster=raster, bands=list(range(1, 100)), percentiles=[(-1, 1)], nodata=None)


def test_clipping_parameter_length():
    """Tests that invalid length for provided parameters raises the correct exception."""
    with pytest.raises(NonMatchingParameterLengthsException):
        with rasterio.open(raster_path) as raster:
            clipping.clipping(raster=raster, bands=[1, 2, 3], limits=[(-1, 1), (-2, 2)], nodata=None)
            clipping.clipping(raster=raster, bands=[1, 2], limits=[(-1, 1), (-2, 2), (-3, 3)], nodata=None)


def test_clipping_percentile_parameter():
    """Tests that invalid percentile values raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        with rasterio.open(raster_path) as raster:
            # All None
            clipping.clipping(raster=raster, bands=None, limits=[(None, None)], nodata=None)


def test_clipping_min_max_position():
    """Tests that invalid min-max positions for provided parameters raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        with rasterio.open(raster_path) as raster:
            # Invalid position of minimum and maximum values for limits
            clipping.clipping(raster=raster, bands=None, limits=[(1, 0)], nodata=None)
            clipping.clipping(raster=raster, bands=[1, 2, 3], limits=[(0, 0), (1, 0), (2, 0)], nodata=None)
