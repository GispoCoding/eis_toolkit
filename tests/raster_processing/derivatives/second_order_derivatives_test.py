from pathlib import Path

import numpy as np
import pytest
import rasterio

from eis_toolkit.exceptions import (
    InvalidParameterValueException,
    InvalidRasterBandException,
    NonSquarePixelSizeException,
)
from eis_toolkit.raster_processing.derivatives.parameters import first_order, second_order_basic_set

parent_dir = Path(__file__).parent.parent
raster_path_single = parent_dir.joinpath("../data/remote/small_raster.tif")
raster_path_multi = parent_dir.joinpath("../data/remote/small_raster_multiband.tif")
raster_path_nonsquared = parent_dir.joinpath("../data/remote/nonsquared_pixelsize_raster.tif")

SECOND_ORDER_BASIC_SET_RESULTS = {
    "Evans": {
        "planc": 0.017252,
        "profc": 0.000793,
    },
    "Young": {
        "planc": 0.018364,
        "profc": 0.001461,
    },
    "Zevenbergen": {
        "planc": 0.012064,
        "profc": 0.000763,
    },
}


@pytest.mark.parametrize("method", ["Evans", "Young", "Zevenbergen"])
def test_second_order_basic_set(method: str):
    """Test the second order basic set functions."""
    with rasterio.open(raster_path_single) as raster:
        # There may be some parameters resulting in zero-pixels,
        # which may influence the test for the slope_tolerance, resulting in an assertion error.
        # Thus, keep it with these derivates for testing.
        parameters = [
            "planc",
            "profc",
        ]

        deriv = second_order_basic_set(raster, parameters=parameters, method=method)

        for parameter in parameters:
            deriv_array = deriv[parameter][0]
            deriv_meta = deriv[parameter][1]

            # Shapes and types
            assert isinstance(deriv_array, np.ndarray)
            assert isinstance(deriv_meta, dict)
            assert deriv_array.shape == (raster.height, raster.width)

            # Check calculated means
            np.testing.assert_almost_equal(
                np.mean(deriv_array), SECOND_ORDER_BASIC_SET_RESULTS[method][parameter], decimal=6
            )

            # Nodata
            test_array = raster.read(1)
            np.testing.assert_array_equal(
                np.ma.masked_values(deriv_array, value=-9999, shrink=False).mask,
                np.ma.masked_values(test_array, value=raster.nodata, shrink=False).mask,
            )

            # Minimum slope applied
            slope_tolerance = 10
            deriv_st = second_order_basic_set(
                raster, parameters=[parameter], method=method, slope_tolerance=slope_tolerance
            )
            deriv_st_array = deriv_st[parameter][0]

            slope = first_order(raster, parameters=["G"], slope_gradient_unit="degrees", method=method)
            slope_array = slope["G"][0]

            np.testing.assert_array_equal(
                np.ma.masked_less_equal(slope_array, value=slope_tolerance).mask,
                np.ma.masked_values(deriv_st_array, value=0, shrink=False).mask,
            )


def test_number_bands():
    """Test if the number of bands is correct."""
    with rasterio.open(raster_path_multi) as raster:
        parameters = ["planc"]
        with pytest.raises(InvalidRasterBandException):
            second_order_basic_set(raster, parameters=parameters)


def test_nonsquared_pixelsize():
    """Test if pixels are squared."""
    with rasterio.open(raster_path_nonsquared) as raster:
        parameters = ["planc"]
        with pytest.raises(NonSquarePixelSizeException):
            second_order_basic_set(raster, parameters=parameters)


def test_scaling():
    """Test if scaling factor is correct."""
    with rasterio.open(raster_path_single) as raster:
        parameters = ["planc"]
        with pytest.raises(InvalidParameterValueException):
            second_order_basic_set(raster, parameters=parameters, scaling_factor=0)
