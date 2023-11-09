import pytest
import rasterio

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.raster_processing.unique_combinations import unique_combinations
from tests.raster_processing.clip_test import raster_path as SMALL_RASTER_PATH

expected_1st_row = [
    675,
    562,
    481,
    432,
    415,
    418,
    393,
    392,
    413,
    433,
    460,
    493,
    518,
    528,
    524,
    523,
    533,
    550,
    573,
    607,
    653,
    694,
    735,
    818,
    867,
    884,
    882,
    905,
    948,
    956,
    941,
    902,
    870,
    808,
    774,
    807,
    845,
    876,
    841,
    737,
    743,
    748,
    752,
    758,
    771,
    782,
]
expected_2nd_row = [
    511,
    440,
    411,
    397,
    384,
    366,
    380,
    400,
    426,
    456,
    499,
    529,
    526,
    521,
    515,
    512,
    539,
    597,
    644,
    686,
    725,
    825,
    886,
    920,
    934,
    938,
    933,
    944,
    966,
    972,
    961,
    946,
    914,
    861,
    800,
    766,
    745,
    741,
    734,
    739,
    749,
    759,
    766,
    777,
    783,
    788,
]


def test_unique_combinations():
    """Test unique combinations creates expected output."""
    with rasterio.open(SMALL_RASTER_PATH) as raster_1, rasterio.open(SMALL_RASTER_PATH) as raster_2:

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
        with rasterio.open(SMALL_RASTER_PATH) as raster:
            unique_combinations([raster])
