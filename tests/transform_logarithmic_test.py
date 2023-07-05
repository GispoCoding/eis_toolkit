import pytest
import rasterio
import numpy as np
from pathlib import Path

from eis_toolkit.transformations import logarithmic
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


def test_log_transform():
    """Test that transformation works as intended"""
    bands = None
    nodata = 3.748
    log_transform_list = ["ln", "log2", "log10"]

    for log_transform in log_transform_list:
        log_transform = [log_transform]

        with rasterio.open(raster_path) as raster:
            out_array, out_meta, out_settings = logarithmic.log_transform(
                raster=raster, bands=bands, log_transform=log_transform, nodata=nodata
            )

            # Output shapes and types
            assert isinstance(out_array, np.ndarray)
            assert isinstance(out_meta, dict)
            assert isinstance(out_settings, dict)

            # Output array (nodata in place)
            test_array = raster.read(list(range(1, out_meta["count"] + 1)))

            np.testing.assert_array_equal(
                np.ma.masked_values(out_array, value=nodata, shrink=False).mask,
                np.logical_or(
                    np.ma.masked_values(test_array, value=nodata, shrink=False).mask,
                    np.ma.masked_less_equal(test_array, 0).mask,
                ),
            )

            # Output array (transformation result)
            log_transform = log_transform * out_meta["count"]
            out_decimals = set_max_precision()
            test_array = cast_array_to_float(test_array, cast_int=True)

            for i in range(0, out_meta["count"]):
                test_array[i] = replace_values(test_array[i], values_to_replace=[nodata, np.inf], replace_value=np.nan)
                test_array[i] = np.where(test_array[i] <= 0, np.nan, test_array[i])

                if log_transform[i] == "ln":
                    test_array[i] = np.log(test_array[i])
                elif log_transform[i] == "log2":
                    test_array[i] = np.log2(test_array[i])
                elif log_transform[i] == "log10":
                    test_array[i] = np.log10(test_array[i])

            test_array = truncate_decimal_places(test_array, decimal_places=out_decimals)
            test_array = nan_to_nodata(test_array, nodata_value=nodata)
            test_array = cast_array_to_float(test_array, scalar=nodata, cast_float=True)

            assert out_array.shape == (out_meta["count"], raster.height, raster.width)
            assert out_array.dtype == test_array.dtype
            np.testing.assert_array_equal(out_array, test_array)


def test_log_transform_band_selection():
    """Tests that invalid parameter value for band selection raises the correct exception."""
    with pytest.raises(InvalidRasterBandException):
        with rasterio.open(raster_path) as raster:
            logarithmic.log_transform(raster=raster, bands=[0], log_transform=["log2"], nodata=None)
            logarithmic.log_transform(raster=raster, bands=list(range(1, 100)), log_transform=["log2"], nodata=None)


def test_log_transform_parameter_length():
    """Tests that invalid length for provided parameters raises the correct exception."""
    with pytest.raises(NonMatchingParameterLengthsException):
        with rasterio.open(raster_path) as raster:
            # Invalid method
            logarithmic.log_transform(raster=raster, bands=[1, 2, 3], log_transform=["log2", "log10"], nodata=None)
            logarithmic.log_transform(raster=raster, bands=[1, 2], log_transform=["ln"], nodata=None)


def test_log_transform_method():
    """Tests that invalid method raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        with rasterio.open(raster_path) as raster:
            # Invalid method
            logarithmic.log_transform(raster=raster, bands=None, log_transform=["python"], nodata=None)
