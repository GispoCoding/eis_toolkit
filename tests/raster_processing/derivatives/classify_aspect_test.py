import numpy as np
import rasterio
import pytest
from pathlib import Path

from eis_toolkit.exceptions import (
    InvalidRasterBandException,
    InvalidParameterValueException,
)
from eis_toolkit.raster_processing.derivatives.classification import classify_aspect
from eis_toolkit.raster_processing.derivatives.parameters import first_order

parent_dir = Path(__file__).parent.parent
raster_path_single = parent_dir.joinpath("../data/remote/small_raster.tif")
raster_path_multi = parent_dir.joinpath("../data/remote/small_raster_multiband.tif")
raster_path_nonsquared = parent_dir.joinpath("../data/remote/nonsquared_pixelsize_raster.tif")

ASPECT_CLASSIFICATION_RESULTS = {
    8: {
        1: 179,
        2: 423,
        3: 206,
        4: 182,
        5: 215,
        6: 483,
        7: 325,
        8: 563,
    },
    16: {
        1: 74,
        2: 83,
        3: 193,
        4: 289,
        5: 71,
        6: 54,
        7: 90,
        8: 137,
        9: 101,
        10: 93,
        11: 236,
        12: 293,
        13: 144,
        14: 193,
        15: 344,
        16: 181,
    },
}


@pytest.mark.parametrize("num_classes", [8, 16])
def test_aspect_classification(num_classes: int):
    with rasterio.open(raster_path_single) as raster:
        parameter = "A"

        aspect = first_order(raster, parameters=[parameter], slope_direction_unit="radians", method="Horn")

        aspect_array = aspect[parameter][0]
        aspect_meta = aspect[parameter][1]

        memory_file = rasterio.MemoryFile()
        with memory_file.open(**aspect_meta) as dst:
            dst.write(aspect_array, 1)

        with memory_file.open() as aspect:
            classification_array, classification_mapping, classification_meta = classify_aspect(
                aspect, num_classes=num_classes, unit="radians"
            )

        # Shapes and types
        assert isinstance(classification_array, np.ndarray)
        assert isinstance(classification_mapping, dict)
        assert isinstance(classification_meta, dict)
        assert classification_array.shape == aspect_array.shape

        # Values and mapping
        assert np.min(classification_array >= -1) and np.max(classification_array <= num_classes)

        if num_classes == 8:
            allowed_classes = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "ND"]
            allowed_values = [1, 2, 3, 4, 5, 6, 7, 8, -1]
        elif num_classes == 16:
            allowed_classes = [
                "N",
                "NNE",
                "NE",
                "ENE",
                "E",
                "ESE",
                "SE",
                "SSE",
                "S",
                "SSW",
                "SW",
                "WSW",
                "W",
                "WNW",
                "NW",
                "NNW",
                "ND",
            ]
            allowed_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, -1]

        for key, value in classification_mapping.items():
            assert key in allowed_classes
            assert value in allowed_values

        # Nodata
        test_array = raster.read(1)
        np.testing.assert_array_equal(
            np.ma.masked_values(classification_array, value=-9999, shrink=False).mask,
            np.ma.masked_values(test_array, value=raster.nodata, shrink=False).mask,
        )

        # Check if the number of pixels in each class is correct
        unique_values, counts = np.unique(classification_array, return_counts=True)
        test_dict = dict(zip(unique_values, counts))
        assert test_dict == ASPECT_CLASSIFICATION_RESULTS[num_classes]


def test_number_bands():
    with rasterio.open(raster_path_multi) as raster:
        with pytest.raises(InvalidRasterBandException):
            classify_aspect(raster)


def test_number_classes():
    with rasterio.open(raster_path_single) as raster:
        with pytest.raises(InvalidParameterValueException):
            classify_aspect(raster, num_classes=7)
