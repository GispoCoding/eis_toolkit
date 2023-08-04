from pathlib import Path

import numpy as np
import pytest
import rasterio

from eis_toolkit.exceptions import (
    InvalidParameterValueException,
    InvalidRasterBandException,
    NonMatchingParameterLengthsException,
)
from eis_toolkit.transformations.linear import (
    _min_max_scaling,
    _z_score_normalization,
    min_max_scaling,
    z_score_normalization,
)
from tests.transformations import check_transformation_outputs

parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("../data/remote/small_raster_multiband.tif")


def test_z_score_normalization():
    """Test that transformation works as intended."""
    bands = None
    nodata = 3.748

    with rasterio.open(raster_path) as raster:
        out_array, out_meta, out_settings = z_score_normalization(raster=raster, bands=bands, nodata=nodata)

        # Output shapes and types
        check_transformation_outputs(out_array, out_meta, out_settings, raster, nodata)

        in_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        transformation, mean, sd = _z_score_normalization(in_array=in_array)
        transformation = transformation.astype(np.float32)
        expected = np.array(
            [
                -1.5811388,
                -1.264911,
                -0.9486833,
                -0.6324555,
                -0.31622776,
                0.0,
                0.31622776,
                0.6324555,
                0.9486833,
                1.2649111,
                1.5811388,
            ],
            dtype=np.float32,
        )
        expected_mean = 5
        expected_sd = 3.1622776601683795

        np.testing.assert_array_equal(transformation, expected)
        assert mean == expected_mean
        assert sd == expected_sd


def test_min_max_scaling():
    """Test that transformation works as intended."""
    bands = None
    nodata = 3.748
    new_range = [(0, 1)]

    with rasterio.open(raster_path) as raster:
        out_array, out_meta, out_settings = min_max_scaling(
            raster=raster, bands=bands, new_range=[(0, 1)], nodata=nodata
        )

        # Output shapes and types
        check_transformation_outputs(out_array, out_meta, out_settings, raster, nodata)

        in_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        transformation = _min_max_scaling(in_array=in_array, new_range=new_range[0])
        transformation = transformation.astype(np.float32)
        expected = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(transformation, expected)


def test_linear_band_selection():
    """Tests that invalid parameter value for band selection raises the correct exception."""
    with pytest.raises(InvalidRasterBandException):
        with rasterio.open(raster_path) as raster:
            z_score_normalization(raster=raster, bands=[0], nodata=None)
            z_score_normalization(raster=raster, bands=list(range(1, 100)), nodata=None)
            min_max_scaling(raster=raster, bands=[0], new_range=[(0, 1)], nodata=None)
            min_max_scaling(raster=raster, bands=list(range(1, 100)), new_range=[(0, 1)], nodata=None)


def test_linear_parameter_length():
    """Tests that invalid length for provided parameters raises the correct exception."""
    with pytest.raises(NonMatchingParameterLengthsException):
        with rasterio.open(raster_path) as raster:
            # Invalid new_range
            min_max_scaling(raster=raster, bands=[1, 2, 3], new_range=[(0, 1), (0, 2)], nodata=None)
            min_max_scaling(raster=raster, bands=[1, 2], new_range=[(0, 1), (0, 2), (0, 3)], nodata=None)


def test_linear_min_max_position():
    """Tests that invalid min-max positions for provided parameters raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        with rasterio.open(raster_path) as raster:
            # Invalid position of minimum and maximum values for new_range
            min_max_scaling(raster=raster, bands=None, new_range=[(1, 0)], nodata=None)
            min_max_scaling(raster=raster, bands=[1, 2, 3], new_range=[(0, 0), (1, 0), (2, 0)], nodata=None)
