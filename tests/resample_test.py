import pytest
import rasterio
import numpy as np
from rasterio import Affine
from pathlib import Path

from eis_toolkit.raster_processing.resampling import resample
from eis_toolkit.exceptions import InvalidParameterValueException, NegativeResamplingFactorException

parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("data/remote/small_raster.tif")


def test_resample():
    """Test that resample function works as intended."""
    src_raster = rasterio.open(raster_path)
    upscale_factor = 2
    _, resampled_meta = resample(src_raster, upscale_factor)

    assert resampled_meta['crs'] == src_raster.meta['crs']
    assert np.array_equal(src_raster.width * upscale_factor, resampled_meta['width'])
    assert np.array_equal(src_raster.height * upscale_factor, resampled_meta['height'])
    assert resampled_meta['transform'] == Affine(src_raster.transform.a / upscale_factor,
                                                 src_raster.transform.b,
                                                 src_raster.transform.c,
                                                 src_raster.transform.d,
                                                 src_raster.transform.e / upscale_factor,
                                                 src_raster.transform.f)


def test_resample_invalid_resampling_method():
    """Tests that invalid parameter value for resampling method raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        with rasterio.open(raster_path) as raster:
            resample(
                raster=raster,
                upscale_factor=2,
                resampling_method=78
            )


def test_resample_negative_upscale_factor():
    """Tests that invalid parameter value for resampling method raises the correct exception."""
    with pytest.raises(NegativeResamplingFactorException):
        with rasterio.open(raster_path) as raster:
            resample(
                raster=raster,
                upscale_factor=-2,
                resampling_method=1
            )