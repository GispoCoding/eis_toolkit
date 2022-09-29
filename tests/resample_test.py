import pytest
import rasterio
import numpy as np
from rasterio import Affine
from pathlib import Path
from rasterio.enums import Resampling


from eis_toolkit.raster_processing.resampling import resample
from eis_toolkit.exceptions import NumericValueSignException

parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("data/remote/small_raster.tif")


def test_resample():
    """Test that resample function works as intended."""
    with rasterio.open(raster_path) as raster:
        upscale_factor = 2
        _, resampled_meta = resample(raster, upscale_factor, Resampling.bilinear)

        assert resampled_meta['crs'] == raster.meta['crs']
        assert np.array_equal(raster.width * upscale_factor, resampled_meta['width'])
        assert np.array_equal(raster.height * upscale_factor, resampled_meta['height'])
        assert resampled_meta['transform'] == Affine(raster.transform.a / upscale_factor,
                                                    raster.transform.b,
                                                    raster.transform.c,
                                                    raster.transform.d,
                                                    raster.transform.e / upscale_factor,
                                                    raster.transform.f)


def test_resample_negative_upscale_factor():
    """Tests that invalid parameter value for resampling method raises the correct exception."""
    with pytest.raises(NumericValueSignException):
        with rasterio.open(raster_path) as raster:
            resample(
                raster=raster,
                upscale_factor=-2,
                resampling_method=Resampling.cubic
            )