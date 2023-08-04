from pathlib import Path

import numpy as np
import pytest
import rasterio

from eis_toolkit.exceptions import (
    InvalidParameterValueException,
    InvalidRasterBandException,
    NonMatchingParameterLengthsException,
)
from eis_toolkit.transformations.sigmoid import _sigmoid_transform, sigmoid_transform
from tests.transformations import check_transformation_outputs

parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("../data/remote/small_raster_multiband.tif")


def test_sigmoid_transform():
    """Test that transformation works as intended."""
    bands = None
    bounds = [(0, 1)]
    slope = [1]
    center = True
    nodata = 3.748

    with rasterio.open(raster_path) as raster:
        out_array, out_meta, out_settings = sigmoid_transform(
            raster=raster, bands=bands, bounds=bounds, slope=slope, center=center, nodata=nodata
        )

        # Output shapes and types
        check_transformation_outputs(out_array, out_meta, out_settings, raster, nodata)


def test_sigmoid_core():
    """Test for core functionality with small example computation."""
    in_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    transformation = _sigmoid_transform(in_array, bounds=(0, 1), slope=1, center=True).astype(np.float32)
    expected = np.array(
        [
            0.00669285,
            0.01798621,
            0.04742587,
            0.11920292,
            0.26894143,
            0.5,
            0.7310586,
            0.8807971,
            0.95257413,
            0.98201376,
            0.9933072,
        ],
        dtype=np.float32,
    )

    np.testing.assert_almost_equal(transformation, expected, decimal=6)


def test_sigmoid_band_selection():
    """Tests that invalid parameter value for band selection raises the correct exception."""
    with pytest.raises(InvalidRasterBandException):
        with rasterio.open(raster_path) as raster:
            sigmoid_transform(raster=raster, bands=[0], bounds=[(0, 1)], slope=[1], center=True, nodata=None)
            sigmoid_transform(
                raster=raster, bands=list(range(1, 100)), bounds=[(0, 1)], slope=[1], center=True, nodata=None
            )


def test_sigmoid_parameter_length():
    """Tests that invalid length for provided parameters raises the correct exception."""
    with pytest.raises(NonMatchingParameterLengthsException):
        with rasterio.open(raster_path) as raster:
            # Invalid bounds
            sigmoid_transform(
                raster=raster, bands=[1, 2, 3], bounds=[(0, 1), (0, 2)], slope=[1], center=True, nodata=None
            )

            # Invalid slope
            sigmoid_transform(raster=raster, bands=[1, 2, 3], bounds=[(0, 1)], slope=[1, 1], center=True, nodata=None)


def test_sigmoid_min_max_position():
    """Tests that invalid min-max positions for provided parameters raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        with rasterio.open(raster_path) as raster:
            # Invalid position of minimum and maximum values for new_range
            sigmoid_transform(raster=raster, bands=None, bounds=[(0, 0)], slope=[1], center=True, nodata=None)
            sigmoid_transform(raster=raster, bands=None, bounds=[(1, 0)], slope=[1], center=True, nodata=None)
