"""Functions tests reproject functionality by reprojecting small_raster to KKJ 
    coordinate system and then comparing it to small_raster which was reprojected to KKJ
    coordinate system with QGIS.
"""
import rasterio
import numpy as np
from eis_toolkit.raster_processing.reprojecting import reproject_raster
from pathlib import Path

parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("data/remote/small_raster.tif")
reprojected_raster_path = parent_dir.joinpath("data/remote/small_raster_qgis_kkj.tif")

src_raster = rasterio.open(raster_path)
reprojected_data, reprojected_meta = reproject_raster(src_raster, 2393)

target_rast = rasterio.open(reprojected_raster_path)
target_data = target_rast.read()
target_meta = target_rast.meta

def test_reproject_data():
    """ This function compares only data. """
    assert np.array_equal(reprojected_data, target_data)

def test_reproject_meta():
    """ This function compares only meta data. """
    assert reprojected_meta['count'] == target_meta['count']
    assert reprojected_meta['crs'] == target_meta['crs']
    assert reprojected_meta['driver'] == target_meta['driver']
    assert reprojected_meta['dtype'] == target_meta['dtype']
    assert reprojected_meta['height'] == target_meta['height']
    assert reprojected_meta['width'] == target_meta['width']
    assert reprojected_meta['transform'] == target_meta['transform']


