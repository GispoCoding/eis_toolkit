from pathlib import Path

import numpy as np
import pytest
import rasterio

from eis_toolkit.exceptions import InvalidRasterBandException, NonSquarePixelSizeException
from eis_toolkit.surface_attributes.slope import get_slope

parent_dir = Path(__file__).parent
raster_path_single = parent_dir.joinpath("../data/remote/small_raster.tif")
raster_path_multi = parent_dir.joinpath("../data/remote/small_raster_multiband.tif")
raster_path_nonsquared = parent_dir.joinpath("../data/remote/nonsquared_pixelsize_raster.tif")


def test_slope():
    """Test that computation works as intended."""

    with rasterio.open(raster_path_single) as raster:
        out_array, out_meta = get_slope(raster, method="Horn81", unit="degrees", scaling_factor=1)

        # Output shapes and types
        assert isinstance(out_array, np.ndarray)
        assert isinstance(out_meta, dict)

        # Output array (nodata in place)
        test_array = raster.read(1)
        np.testing.assert_array_equal(
            np.ma.masked_values(out_array, value=-9999, shrink=False).mask,
            np.ma.masked_values(test_array, value=raster.nodata, shrink=False).mask,
        )


def test_slope_band_selection():
    """Tests that invalid number of raste bands raise correct exception."""
    with pytest.raises(InvalidRasterBandException):
        with rasterio.open(raster_path_multi) as raster:
            out_array, out_meta = get_slope(raster)


def test_slope_nonsquare_pixelsize():
    """Tests that raster with non-squared pixels raise correct exception."""
    with pytest.raises(NonSquarePixelSizeException):
        with rasterio.open(raster_path_nonsquared) as raster:
            out_array, out_meta = get_slope(raster)
