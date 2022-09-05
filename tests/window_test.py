from operator import le
import numpy as np
import rasterio
import pytest
from eis_toolkit.raster_processing.windowing import extract_window
from eis_toolkit.exceptions import InvalidWindowSizeException
from eis_toolkit.exceptions import CoordinatesOutOfBoundExeption
from pathlib import Path
from rasterio.coords import BoundingBox

parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("data/remote/small_raster.tif")
raster = rasterio.open(raster_path)

# Values from corresponding operation executed in QGIS for 3x3 window, small raster as
# source raster. Following center coordinates were used:
easting = 384747
northing = 6671293
target_data = np.array([[[3.242, 3.044, 2.870],
                        [3.316, 3.136, 2.983],
                        [3.390, 3.232, 3.065]]])
target_transform = rasterio.Affine(
        2.0,
        0.0,
        384744.0,
        0.0,
        -2.0,
        6671296.0
)
target_meta = raster.meta.copy()
target_meta.update({
    'height': 3,
    'width': 3,
    'transform': target_transform
})

window_data, window_meta = extract_window(
    raster=raster,
    center_x=easting,
    center_y=northing,
    win_size=3
)

def test_window_data():
    """ This function compares only data."""
    assert np.array_equal(window_data, target_data)

def test_window_meta():
    """ This function compares only meta data."""
    assert window_meta['count'] == target_meta['count']
    assert window_meta['crs'] == target_meta['crs']
    assert window_meta['driver'] == target_meta['driver']
    assert window_meta['dtype'] == target_meta['dtype']
    assert window_meta['height'] == target_meta['height']
    assert window_meta['width'] == target_meta['width']
    assert window_meta['transform'] == target_meta['transform']

def test_invalid_window():
    """Test that invalid window size raises correct exeption"""
    with pytest.raises(InvalidWindowSizeException):
        extract_window(
        raster=raster,
        center_x=easting,
        center_y=northing,
        win_size=2
        )

def test_out_of_bounds_coordinates():
    """Test that out of bound coordinates raises correct exeption"""
    with pytest.raises(CoordinatesOutOfBoundExeption):
        extract_window(
        raster=raster,
        center_x=100,
        center_y=100,
        win_size=9
        )   
