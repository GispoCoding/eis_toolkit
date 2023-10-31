from pathlib import Path

import numpy as np
import pytest
import rasterio

from eis_toolkit.exceptions import (
    InvalidParameterValueException,
    InvalidRasterBandException,
    NonMatchingParameterLengthsException,
)
from eis_toolkit.transformations.clip import _clip_transform, clip_transform
from tests.transformations import check_transformation_outputs

parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("../data/remote/small_raster_multiband.tif")


def test_clip_transform():
    """Test that transformation works as intended."""
    bands = None
    limits = [(-1, 1)]

    nodata = 3.748

    with rasterio.open(raster_path) as raster:
        out_array, out_meta, out_settings = clip_transform(raster=raster, bands=bands, limits=limits, nodata=nodata)

        # Output shapes and types
        check_transformation_outputs(out_array, out_meta, out_settings, raster, nodata)


def test_clip_transform_core():
    """Test for core functionality with small example computation."""
    in_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    transformation = _clip_transform(in_array, limits=(2, 8))
    expected = np.array([2, 2, 2, 3, 4, 5, 6, 7, 8, 8, 8])

    np.testing.assert_array_equal(transformation, expected)


def test_clip_transform_band_selection():
    """Tests that invalid parameter value for band selection raises the correct exception."""
    with pytest.raises(InvalidRasterBandException):
        with rasterio.open(raster_path) as raster:
            clip_transform(raster=raster, bands=[0], limits=[(-1, 1)], nodata=None)
            clip_transform(raster=raster, bands=list(range(1, 100)), percentiles=[(-1, 1)], nodata=None)


def test_clip_transform_parameter_length():
    """Tests that invalid length for provided parameters raises the correct exception."""
    with pytest.raises(NonMatchingParameterLengthsException):
        with rasterio.open(raster_path) as raster:
            clip_transform(raster=raster, bands=[1, 2, 3], limits=[(-1, 1), (-2, 2)], nodata=None)
            clip_transform(raster=raster, bands=[1, 2], limits=[(-1, 1), (-2, 2), (-3, 3)], nodata=None)


def test_clip_transform_percentile_parameter():
    """Tests that invalid percentile values raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        with rasterio.open(raster_path) as raster:
            # All None
            clip_transform(raster=raster, bands=None, limits=[(None, None)], nodata=None)


def test_clip_transform_min_max_position():
    """Tests that invalid min-max positions for provided parameters raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        with rasterio.open(raster_path) as raster:
            # Invalid position of minimum and maximum values for limits
            clip_transform(raster=raster, bands=None, limits=[(1, 0)], nodata=None)
            clip_transform(raster=raster, bands=[1, 2, 3], limits=[(0, 0), (1, 0), (2, 0)], nodata=None)
