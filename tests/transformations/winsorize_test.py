from pathlib import Path

import numpy as np
import pytest
import rasterio

from eis_toolkit.exceptions import (
    InvalidParameterValueException,
    InvalidRasterBandException,
    NonMatchingParameterLengthsException,
)
from eis_toolkit.transformations.winsorize import _winsorize, winsorize
from tests.transformations import check_transformation_outputs

parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("../data/remote/small_raster_multiband.tif")


def test_winsorizing():
    """Test that transformation works as intended."""
    bands = None
    percentiles = [(10, 10)]
    inside = False

    nodata = 3.748

    with rasterio.open(raster_path) as raster:
        out_array, out_meta, out_settings = winsorize(
            raster=raster, bands=bands, percentiles=percentiles, inside=inside, nodata=nodata
        )

        # Output shapes and types
        check_transformation_outputs(out_array, out_meta, out_settings, raster, nodata)


def test_winsorize_core():
    """Test for core functionality with small example computation."""
    in_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    transformation, lower, upper = _winsorize(in_array, percentiles=(10, 10), inside=True)
    expected = np.array([1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9])
    expected_lower = 1
    expected_upper = 9

    np.testing.assert_array_equal(transformation, expected)
    assert lower == expected_lower
    assert upper == expected_upper


def test_winsorize_band_selection():
    """Tests that invalid parameter value for band selection raises the correct exception."""
    with pytest.raises(InvalidRasterBandException):
        with rasterio.open(raster_path) as raster:
            winsorize(raster=raster, bands=[0], percentiles=[(10, 10)], inside=False, nodata=None)
            winsorize(raster=raster, bands=list(range(1, 100)), percentiles=[(10, 10)], inside=False, nodata=None)


def test_winsorize_parameter_length():
    """Tests that invalid length for provided parameters raises the correct exception."""
    with pytest.raises(NonMatchingParameterLengthsException):
        with rasterio.open(raster_path) as raster:
            winsorize(raster=raster, bands=[1, 2, 3], percentiles=[(10, 10), (10, 10)], inside=False, nodata=None)
            winsorize(
                raster=raster, bands=[1, 2], percentiles=[(10, 10), (10, 10), (10, 10)], inside=False, nodata=None
            )


def test_winsorize_percentile_parameter():
    """Tests that invalid percentile values raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        with rasterio.open(raster_path) as raster:
            # All None
            winsorize(raster=raster, bands=None, percentiles=[(None, None)], inside=False, nodata=None)

            # Sum > 100
            winsorize(raster=raster, bands=None, percentiles=[(60, 40)], inside=False, nodata=None)

            # Invalid lower value
            winsorize(raster=raster, bands=None, percentiles=[(100, None)], inside=False, nodata=None)

            # Invalid upper value
            winsorize(raster=raster, bands=None, percentiles=[(None, 100)], inside=False, nodata=None)
