import pytest
import rasterio
import numpy as np
from pathlib import Path

from eis_toolkit.transformations import binarize
from eis_toolkit.utilities.miscellaneous import cast_scalar_to_int, check_dtype_for_int
from eis_toolkit.exceptions import InvalidRasterBandException, NonMatchingParameterLengthsException

parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("data/remote/small_raster_multiband.tif")


def test_binarize():
    """Test that transformation works as intended"""
    bands = None
    thresholds = [2]
    nodata = 3.748
    nodata = cast_scalar_to_int(rasterio.open(raster_path).nodata if nodata is None else nodata)

    with rasterio.open(raster_path) as raster:
        out_array, out_meta, out_settings = binarize.binarize(
            raster=raster, bands=bands, thresholds=thresholds, nodata=nodata
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
        thresholds = thresholds * out_meta["count"]
        initial_dtype = test_array.dtype

        for i in range(0, out_meta["count"]):
            test_mask = np.ma.masked_values(test_array[i], value=nodata, shrink=False).mask
            test_array[i] = np.where(test_array[i] <= thresholds[i], 0, 1)
            test_array[i] = np.where(test_mask, nodata, test_array[i])

        if not check_dtype_for_int(nodata):
            test_array = test_array.astype(initial_dtype)
        else:
            test_array = test_array.astype(np.min_scalar_type(nodata))

        assert out_array.shape == (out_meta["count"], raster.height, raster.width)
        assert out_array.dtype == test_array.dtype
        np.testing.assert_array_equal(out_array, test_array)


def test_binarize_band_selection():
    """Tests that invalid parameter value for band selection raises the correct exception."""
    with pytest.raises(InvalidRasterBandException):
        with rasterio.open(raster_path) as raster:
            binarize.binarize(raster=raster, bands=[0], thresholds=[2], nodata=None)
            binarize.binarize(raster=raster, bands=list(range(1, 100)), thresholds=[2], nodata=None)


def test_binarize_parameter_length():
    """Tests that invalid length for provided parameters raises the correct exception."""
    with pytest.raises(NonMatchingParameterLengthsException):
        with rasterio.open(raster_path) as raster:
            # Invalid Threshold
            binarize.binarize(raster=raster, bands=[1, 2, 3], thresholds=[1, 2], nodata=None)
            binarize.binarize(raster=raster, bands=[1, 2], thresholds=[1, 2, 3], nodata=None)
