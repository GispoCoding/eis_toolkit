from pathlib import Path

import numpy as np
import pytest
import rasterio

from eis_toolkit.exceptions import (
    InvalidParameterValueException,
    InvalidRasterBandException,
    NonMatchingParameterLengthsException,
)
from eis_toolkit.transformations import winsorize
from eis_toolkit.utilities.miscellaneous import cast_array_to_float, cast_array_to_int, cast_scalar_to_int
from eis_toolkit.utilities.nodata import nan_to_nodata, nodata_to_nan

parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("data/remote/small_raster_multiband.tif")


def test_winsorizing():
    """Test that transformation works as intended."""
    bands = None
    percentiles = [(10, 10)]
    inside = False

    nodata = 3.748
    nodata = cast_scalar_to_int(rasterio.open(raster_path).nodata if nodata is None else nodata)

    with rasterio.open(raster_path) as raster:
        out_array, out_meta, out_settings = winsorize.winsorize(
            raster=raster, bands=bands, percentiles=percentiles, inside=inside, nodata=nodata
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
        percentiles = percentiles * out_meta["count"]
        initial_dtype = test_array.dtype
        test_array = cast_array_to_float(test_array, cast_int=True)

        for i in range(0, out_meta["count"]):
            test_array[i] = nodata_to_nan(test_array[i], nodata_value=nodata)

            clean_array = np.extract(np.isfinite(test_array[i]), test_array[i])

            if inside is True:
                lower, upper = "lower", "higher"
            else:
                lower, upper = "higher", "lower"

            percentile_lower = np.percentile(clean_array, percentiles[i][0], method=lower)
            percentile_upper = np.percentile(clean_array, 100 - percentiles[i][1], method=upper)

            test_array[i] = np.where(test_array[i] < percentile_lower, percentile_lower, test_array[i])
            test_array[i] = np.where(test_array[i] > percentile_upper, percentile_upper, test_array[i])

        test_array = nan_to_nodata(test_array, nodata_value=nodata)
        test_array = cast_array_to_int(test_array, scalar=nodata, initial_dtype=initial_dtype)

        assert out_array.shape == (out_meta["count"], raster.height, raster.width)
        assert out_array.dtype == test_array.dtype
        np.testing.assert_array_equal(out_array, test_array)


def test_winsorize_band_selection():
    """Tests that invalid parameter value for band selection raises the correct exception."""
    with pytest.raises(InvalidRasterBandException):
        with rasterio.open(raster_path) as raster:
            winsorize.winsorize(raster=raster, bands=[0], percentiles=[(10, 10)], inside=False, nodata=None)
            winsorize.winsorize(
                raster=raster, bands=list(range(1, 100)), percentiles=[(10, 10)], inside=False, nodata=None
            )


def test_winsorize_parameter_length():
    """Tests that invalid length for provided parameters raises the correct exception."""
    with pytest.raises(NonMatchingParameterLengthsException):
        with rasterio.open(raster_path) as raster:
            winsorize.winsorize(
                raster=raster, bands=[1, 2, 3], percentiles=[(10, 10), (10, 10)], inside=False, nodata=None
            )
            winsorize.winsorize(
                raster=raster, bands=[1, 2], percentiles=[(10, 10), (10, 10), (10, 10)], inside=False, nodata=None
            )


def test_winsorize_percentile_parameter():
    """Tests that invalid percentile values raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        with rasterio.open(raster_path) as raster:
            # All None
            winsorize.winsorize(raster=raster, bands=None, percentiles=[(None, None)], inside=False, nodata=None)

            # Sum > 100
            winsorize.winsorize(raster=raster, bands=None, percentiles=[(60, 40)], inside=False, nodata=None)

            # Invalid lower value
            winsorize.winsorize(raster=raster, bands=None, percentiles=[(100, None)], inside=False, nodata=None)

            # Invalid upper value
            winsorize.winsorize(raster=raster, bands=None, percentiles=[(None, 100)], inside=False, nodata=None)
