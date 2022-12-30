
from pathlib import Path

from typing import Tuple
import numpy as np
import rasterio

from rasterio.enums import Resampling

# Aktuell Testrahmen des Einlesens eines GRID 
# Dateiname eines GRID: 
parent_dir = Path(__file__).parent.parent.parent
name_tif = parent_dir.joinpath(r'tests/dat/remote/small_raster.tif')
#name_tif = r'D:\Projekte\_EIS\Daten\EIS_data\EIS_IOCG_target_area_CLB\Primary_data\Mag\IOCG_Mag_grysc_DGRF65_anom_.tif'



raster = rasterio.open(name_tif)
#raster = rasterio.io.DatasetReader(name_tif).read

# def _test_function(
#     a: int,
#     b: int,
# ) -> int:
#     """Test whether simple function of summing two values together works.

#     Args:
#         a (int): first input integer.
#         b (int): second input integer.

#     Returns:
#         int: sum of the input values.
#     """
#     return a + b
    
# def test_function(
#     a: int,
#     b: int,
# ) -> int:
#     """Test whether simple function of subtracting 5 from input value works.

#     Args:
#         a (int): input integer.
#         b (int): second input integer.
#     Returns:
#         int: result after the subtraction operation.
#     """
#     return _test_function (a,b)


# print(test_function(1,2))
# print(1)


