import pytest
import rasterio
from rasterio import Affine
from rasterio.enums import Resampling

from eis_toolkit.exceptions import NumericValueSignException
from eis_toolkit.raster_processing.resampling import resample
from tests.raster_processing.clip_test import raster_path as SMALL_RASTER_PATH


def test_resample():
    """Test that resample function works as intended."""
    with rasterio.open(SMALL_RASTER_PATH) as raster:
        target_resolution = 6
        _, resampled_meta = resample(raster, target_resolution, resampling_method=Resampling.bilinear)

        assert resampled_meta["crs"] == raster.meta["crs"]
        assert resampled_meta["transform"] == Affine(
            target_resolution,
            raster.transform.b,
            raster.transform.c,
            raster.transform.d,
            -target_resolution,
            raster.transform.f,
        )


def test_resample_negative_upscale_factor():
    """Tests that invalid parameter value for resampling method raises the correct exception."""
    with pytest.raises(NumericValueSignException):
        with rasterio.open(SMALL_RASTER_PATH) as raster:
            resample(raster=raster, resolution=-2, resampling_method=Resampling.cubic)
