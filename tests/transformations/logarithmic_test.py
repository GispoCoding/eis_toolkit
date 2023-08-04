from pathlib import Path

import numpy as np
import pytest
import rasterio

from eis_toolkit.exceptions import (
    InvalidParameterValueException,
    InvalidRasterBandException,
    NonMatchingParameterLengthsException,
)
from eis_toolkit.transformations.logarithmic import (
    _log_transform_ln,
    _log_transform_log2,
    _log_transform_log10,
    log_transform,
)
from tests.transformations import check_transformation_outputs

parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("../data/remote/small_raster_multiband.tif")


def test_log_transform():
    """Test that transformation works as intended."""
    bands = None
    nodata = 3.748

    with rasterio.open(raster_path) as raster:
        for method in ["ln"]:

            out_array, out_meta, out_settings = log_transform(
                raster=raster, bands=bands, log_transform=[method], nodata=nodata
            )

            # Output shapes and types
            check_transformation_outputs(out_array, out_meta, out_settings, raster, nodata, test=2)

        in_array = np.array([np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        transformation_ln = _log_transform_ln(in_array).astype(np.float32)
        expected_ln = np.array(
            [
                np.nan,
                0.0,
                0.6931472,
                1.0986123,
                1.3862944,
                1.609438,
                1.7917595,
                1.9459101,
                2.0794415,
                2.1972246,
                2.3025851,
            ],
            dtype=np.float32,
        )

        transformation_log2 = _log_transform_log2(in_array).astype(np.float32)
        expected_log2 = np.array(
            [np.nan, 0.0, 1.0, 1.5849625, 2.0, 2.321928, 2.5849626, 2.807355, 3.0, 3.169925, 3.321928], dtype=np.float32
        )

        transformation_log10 = _log_transform_log10(in_array).astype(np.float32)
        expected_log10 = np.array(
            [np.nan, 0.0, 0.30103, 0.47712126, 0.60206, 0.69897, 0.7781513, 0.845098, 0.90309, 0.9542425, 1.0],
            dtype=np.float32,
        )

        np.testing.assert_array_equal(transformation_ln, expected_ln)
        np.testing.assert_array_equal(transformation_log2, expected_log2)
        np.testing.assert_array_equal(transformation_log10, expected_log10)


def test_log_transform_band_selection():
    """Tests that invalid parameter value for band selection raises the correct exception."""
    with pytest.raises(InvalidRasterBandException):
        with rasterio.open(raster_path) as raster:
            log_transform(raster=raster, bands=[0], log_transform=["log2"], nodata=None)
            log_transform(raster=raster, bands=list(range(1, 100)), log_transform=["log2"], nodata=None)


def test_log_transform_parameter_length():
    """Tests that invalid length for provided parameters raises the correct exception."""
    with pytest.raises(NonMatchingParameterLengthsException):
        with rasterio.open(raster_path) as raster:
            # Invalid method
            log_transform(raster=raster, bands=[1, 2, 3], log_transform=["log2", "log10"], nodata=None)
            log_transform(raster=raster, bands=[1, 2], log_transform=["ln"], nodata=None)


def test_log_transform_method():
    """Tests that invalid method raises the correct exception."""
    with pytest.raises(InvalidParameterValueException):
        with rasterio.open(raster_path) as raster:
            # Invalid method
            log_transform(raster=raster, bands=None, log_transform=["python"], nodata=None)
