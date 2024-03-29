from pathlib import Path

import numpy as np
import pytest
import rasterio

from eis_toolkit.exceptions import (
    InvalidParameterValueException,
    InvalidRasterBandException,
    NonSquarePixelSizeException,
)
from eis_toolkit.raster_processing.derivatives.parameters import first_order

parent_dir = Path(__file__).parent.parent
raster_path_single = parent_dir.joinpath("../data/remote/small_raster.tif")
raster_path_multi = parent_dir.joinpath("../data/remote/small_raster_multiband.tif")
raster_path_nonsquared = parent_dir.joinpath("../data/remote/nonsquared_pixelsize_raster.tif")

FIRST_ORDER_RESULTS = {
    "Horn": {"A": 200.727, "G": 7.074},
    "Evans": {"A": 200.961, "G": 7.017},
    "Young": {"A": 200.961, "G": 7.017},
    "Zevenbergen": {"A": 200.929, "G": 7.282},
}

FIRST_ORDER_RESULTS_MIN_SLOPE = {"Evans": {"A": 0.396}}


@pytest.mark.parametrize("method", ["Horn", "Evans", "Young", "Zevenbergen"])
def test_first_order(method: str):
    """Test the first order derivative functions."""
    with rasterio.open(raster_path_single) as raster:
        parameters = ["A", "G"]

        deriv = first_order(
            raster,
            parameters=parameters,
            slope_gradient_unit="degrees",
            slope_direction_unit="degrees",
            method=method,
        )

        for parameter in parameters:
            deriv_array = deriv[parameter][0]
            deriv_meta = deriv[parameter][1]

            # Shapes and types
            assert isinstance(deriv_array, np.ndarray)
            assert isinstance(deriv_meta, dict)
            assert deriv_array.shape == (raster.height, raster.width)

            # Value range
            if parameter == "A":
                assert np.min(deriv_array) >= 0 and np.max(deriv_array) <= 360
            elif parameter == "G":
                assert np.min(deriv_array) >= 0 and np.max(deriv_array) <= 90

            # Nodata
            test_array = raster.read(1)
            np.testing.assert_array_equal(
                np.ma.masked_values(deriv_array, value=-9999, shrink=False).mask,
                np.ma.masked_values(test_array, value=raster.nodata, shrink=False).mask,
            )

            # Check calculated means
            np.testing.assert_almost_equal(np.mean(deriv_array), FIRST_ORDER_RESULTS[method][parameter], decimal=3)

            # Run with minimum slope applied for aspect
            if parameter == "A":
                aspect = first_order(
                    raster, parameters=["A"], slope_gradient_unit="degrees", slope_tolerance=10, method=method
                )
                aspect_array = aspect[parameter][0]

                # Check calculated mean (only for one method)
                if method == "Evans":
                    np.testing.assert_almost_equal(
                        np.mean(aspect_array), FIRST_ORDER_RESULTS_MIN_SLOPE[method][parameter], decimal=3
                    )

                slope = first_order(raster, parameters=["G"], slope_gradient_unit="degrees", method=method)
                slope_array = slope["G"][0]

                np.testing.assert_array_equal(
                    np.ma.masked_less_equal(slope_array, value=10).mask,
                    np.ma.masked_values(aspect_array, value=-1, shrink=False).mask,
                )


def test_number_bands():
    """Test if the number of bands is correct."""
    with rasterio.open(raster_path_multi) as raster:
        parameters = ["A", "G"]
        with pytest.raises(InvalidRasterBandException):
            first_order(raster, parameters=parameters)


def test_nonsquared_pixelsize():
    """Test if pixels are squared."""
    with rasterio.open(raster_path_nonsquared) as raster:
        parameters = ["A", "G"]
        with pytest.raises(NonSquarePixelSizeException):
            first_order(raster, parameters=parameters)


def test_scaling():
    """Test if scaling factor is correct.""" ""
    with rasterio.open(raster_path_single) as raster:
        parameters = ["A", "G"]
        with pytest.raises(InvalidParameterValueException):
            first_order(raster, parameters=parameters, scaling_factor=0)
