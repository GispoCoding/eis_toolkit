import numpy as np
import rasterio
import pytest
from pathlib import Path

from eis_toolkit.exceptions import (
    InvalidRasterBandException,
    InvalidParameterValueException,
)
from eis_toolkit.surface_attributes.classification import classify_aspect
from eis_toolkit.surface_attributes.parameters import first_order

parent_dir = Path(__file__).parent
raster_path_single = parent_dir.joinpath("../data/remote/small_raster.tif")
raster_path_multi = parent_dir.joinpath("../data/remote/small_raster_multiband.tif")
raster_path_nonsquared = parent_dir.joinpath("../data/remote/nonsquared_pixelsize_raster.tif")


def test_aspect_classification():
    
    for num_classes in [8, 16]:
      with rasterio.open(raster_path_single) as raster:    
        parameter = "A"
        
        aspect = first_order(
            raster,
            parameters=[parameter],
            slope_direction_unit="degrees",
            method="Horn"
        )

        aspect_array = aspect[parameter][0]
        aspect_meta = aspect[parameter][1]
        
        memory_file = rasterio.MemoryFile()
        with memory_file.open(**aspect_meta) as dst:
          dst.write(aspect_array, 1)

        with memory_file.open() as aspect:
          classification_array, classification_mapping, classification_meta = classify_aspect(aspect, num_classes=8)

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
          allowed_classes = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "ND"]
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


def test_number_bands():
  with rasterio.open(raster_path_multi) as raster:
      with pytest.raises(InvalidRasterBandException):
          classify_aspect(raster)


def test_number_classes():
  with rasterio.open(raster_path_single) as raster:
    with pytest.raises(InvalidParameterValueException):
      classify_aspect(raster, num_classes=7)
      