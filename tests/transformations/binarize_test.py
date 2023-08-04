from pathlib import Path

import numpy as np
import pytest
import rasterio

from eis_toolkit.exceptions import InvalidRasterBandException, NonMatchingParameterLengthsException
from eis_toolkit.transformations.binarize import _binarize, binarize
from tests.transformations import check_transformation_outputs

parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("../data/remote/small_raster_multiband.tif")


def test_binarize():
    """Test that transformation works as intended."""
    bands = None
    thresholds = [2]
    nodata = 3.748

    with rasterio.open(raster_path) as raster:
        out_array, out_meta, out_settings = binarize(raster=raster, bands=bands, thresholds=thresholds, nodata=nodata)

        # Output shapes and types
        check_transformation_outputs(out_array, out_meta, out_settings, raster, nodata)


def test_binarize_core():
    """Test for core functionality with small example computation."""
    in_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    transformation = _binarize(in_array=in_array, threshold=2)
    expected = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

    np.testing.assert_array_equal(transformation, expected)


def test_binarize_band_selection():
    """Tests that invalid parameter value for band selection raises the correct exception."""
    with pytest.raises(InvalidRasterBandException):
        with rasterio.open(raster_path) as raster:
            binarize(raster=raster, bands=[0], thresholds=[2], nodata=None)
            binarize(raster=raster, bands=list(range(1, 100)), thresholds=[2], nodata=None)


def test_binarize_parameter_length():
    """Tests that invalid length for provided parameters raises the correct exception."""
    with pytest.raises(NonMatchingParameterLengthsException):
        with rasterio.open(raster_path) as raster:
            # Invalid Threshold
            binarize(raster=raster, bands=[1, 2, 3], thresholds=[1, 2], nodata=None)
            binarize(raster=raster, bands=[1, 2], thresholds=[1, 2, 3], nodata=None)
