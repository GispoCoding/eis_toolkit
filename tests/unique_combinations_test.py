from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio

from eis_toolkit.raster_processing.unique_combinations import unique_combinations
from eis_toolkit.exceptions import InvalidParameterValueException

parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("data/remote/small_raster.tif")
expected_1st_row = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0]
expected_2nd_row = [47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0,
63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0,
83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 82.0, 89.0, 90.0, 91.0]

def test_unique_combinations():
    """Test unique combinations creates expected output."""
    with rasterio.open(raster_path) as raster_1, rasterio.open(raster_path) as raster_2:

        out_image, out_meta = unique_combinations([raster_1, raster_2])

        assert out_meta["count"] == 1
        assert out_meta["crs"] == raster_1.meta["crs"]
        assert out_meta["driver"] == raster_1.meta["driver"]
        assert out_meta["dtype"] == raster_1.meta["dtype"]
        assert out_meta["height"] == raster_1.meta["height"]
        assert out_meta["width"] == raster_1.meta["width"]

        assert out_image[0].tolist() == expected_1st_row
        assert out_image[1].tolist() == expected_2nd_row


def test_unique_combinations_invalid_parameter():
    """Test that invalid parameter values for rasters raises correct exception."""
    with pytest.raises(InvalidParameterValueException):
        with rasterio.open(raster_path) as raster:
            unique_combinations([raster])
