from pathlib import Path

import numpy as np
import rasterio
from rasterio import profiles

from eis_toolkit.utilities.raster import (
    combine_raster_bands,
    profile_from_extent_and_pixel_size,
    split_raster_bands,
    stack_raster_arrays,
)
from tests.raster_processing.clip_test import raster_path as SMALL_RASTER_PATH

TESTS_DIR = Path(__file__).parent.parent
MULTIBAND_RASTER_PATH = TESTS_DIR.joinpath("data/remote/small_raster_multiband.tif")


def test_split_raster_bands():
    """Test that splitting a multiband raster into singleband raster works as expected."""
    with rasterio.open(MULTIBAND_RASTER_PATH) as raster:
        out_rasters = split_raster_bands(raster)

    assert len(out_rasters) > 1
    for out_image, out_profile in out_rasters:
        assert isinstance(out_image, np.ndarray)
        assert out_image.ndim == 2
        assert isinstance(out_profile, profiles.Profile)
        assert out_profile["count"] == 1


def test_combine_raster_bands():
    """Test that combining multiple rasters into one works as expected."""
    with rasterio.open(SMALL_RASTER_PATH) as raster_1:
        with rasterio.open(SMALL_RASTER_PATH) as raster_2:
            out_image, out_profile = combine_raster_bands([raster_1, raster_2])

    assert out_image.ndim == 3
    assert len(out_image) == 2
    assert isinstance(out_profile, profiles.Profile)
    assert out_profile["count"] == 2


def test_stack_raster_arrays():
    """Test that stacking raster arrays works as expected."""
    with rasterio.open(SMALL_RASTER_PATH) as raster_1:
        arr_1 = raster_1.read(1)
        with rasterio.open(SMALL_RASTER_PATH) as raster_2:
            arr_2, _ = combine_raster_bands([raster_1, raster_2])
    stacked_arrays = stack_raster_arrays([arr_1, arr_2])

    assert len(stacked_arrays) == 3  # Combined singleband raster and multiband (2 bands) so should be 3


def test_profile_from_extent_and_pixel_size():
    """Test that creating raster profile from extent and pixel size works as expected."""
    with rasterio.open(SMALL_RASTER_PATH) as small_raster:
        profile = small_raster.profile.copy()
        pixel_size = profile["transform"].a
        west, south, east, north = small_raster.bounds

    raster_profile = profile_from_extent_and_pixel_size((west, east, south, north), pixel_size)

    np.testing.assert_equal(raster_profile["height"], profile["height"])
    np.testing.assert_equal(raster_profile["width"], profile["width"])
    np.testing.assert_equal(raster_profile["transform"], profile["transform"])
