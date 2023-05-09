import numpy as np
import pytest
import rasterio

from eis_toolkit.exceptions import CoordinatesOutOfBoundsException, InvalidWindowSizeException
from eis_toolkit.raster_processing.windowing import extract_window
from tests.raster_processing.clip_test import raster_path as SMALL_RASTER_PATH

raster = rasterio.open(SMALL_RASTER_PATH)

# Reference values acuired from QGIS from small_raster.tif
# Case1 3x3 window where part of the window is out of bounds
case1_easting = 384745
case1_northing = 6671273
case1_reference_data = np.array([[[-9999, 3.105, 3.152], [-9999, 3.004, 3.037], [-9999, -9999, -9999]]])
case1_reference_transform = rasterio.Affine(2.0, 0.0, 384742.0, 0.0, -2.0, 6671276.0)
case1_reference_meta = raster.meta.copy()
case1_reference_meta.update({"height": 3, "width": 3, "transform": case1_reference_transform})

# Case2 2x2 window
case2_easting = 384781.5
case2_northing = 6671361.5
case2_reference_data = np.array([[[4.827, 4.812], [3.683, 3.318]]])
case2_reference_transform = rasterio.Affine(2.0, 0.0, 384780.0, 0.0, -2.0, 6671364.0)
case2_reference_meta = raster.meta.copy()
case2_reference_meta.update({"height": 2, "width": 2, "transform": case2_reference_transform})

# Case3 3x2 window where part of the window is out of bounds
case3_easting = 384804.19
case3_northing = 6671272.15
case3_reference_data = np.array([[[7.502, 8.459], [6.788, 7.831], [-9999, -9999]]])
case3_reference_transform = rasterio.Affine(2.0, 0.0, 384802.0, 0.0, -2.0, 6671276.0)
case3_reference_meta = raster.meta.copy()
case3_reference_meta.update({"height": 3, "width": 2, "transform": case3_reference_transform})


def test_extract_window_case1():
    """Tests extract_window function in Case1."""
    case1_window_data, case1_window_meta = extract_window(
        raster=raster, center_coords=(case1_easting, case1_northing), height=3, width=3
    )
    assert np.array_equal(case1_window_data, case1_reference_data)
    assert case1_window_meta["height"] == case1_reference_meta["height"]
    assert case1_window_meta["width"] == case1_reference_meta["width"]
    assert case1_window_meta["transform"] == case1_reference_meta["transform"]


def test_extract_window_case2():
    """Tests extract_window function in Case2."""
    case2_window_data, case2_window_meta = extract_window(
        raster=raster, center_coords=(case2_easting, case2_northing), height=2, width=2
    )
    assert np.array_equal(case2_window_data, case2_reference_data)
    assert case2_window_meta["height"] == case2_reference_meta["height"]
    assert case2_window_meta["width"] == case2_reference_meta["width"]
    assert case2_window_meta["transform"] == case2_reference_meta["transform"]


def test_extract_window_case3():
    """Tests extract_window function in Case3."""
    case3_window_data, case3_window_meta = extract_window(
        raster=raster, center_coords=(case3_easting, case3_northing), height=3, width=2
    )
    assert np.array_equal(case3_window_data, case3_reference_data)
    assert case3_window_meta["height"] == case3_reference_meta["height"]
    assert case3_window_meta["width"] == case3_reference_meta["width"]
    assert case3_window_meta["transform"] == case3_reference_meta["transform"]


def test_extract_window_invalid_window():
    """Test that invalid window size raises correct exception."""
    with pytest.raises(InvalidWindowSizeException):
        extract_window(raster=raster, center_coords=(case1_easting, case1_northing), height=0, width=0)


def test_extract_window_out_of_bounds_coordinates():
    """Test that out of bound coordinates raises correct exception."""
    with pytest.raises(CoordinatesOutOfBoundsException):
        extract_window(raster=raster, center_coords=(100, 100), height=1, width=1)
