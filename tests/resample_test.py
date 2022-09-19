
"""Functions tests reproject functionality by reprojecting small_raster to KKJ 
    coordinate system and then comparing it to small_raster which was reprojected to KKJ
    coordinate system with QGIS.
"""
import rasterio
from rasterio import Affine
import numpy as np
from eis_toolkit.raster_processing.resampling import resample_raster
from pathlib import Path

parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("data/remote/small_raster.tif")

src_raster = rasterio.open(raster_path)
src_width = src_raster.width
src_height = src_raster.height

upscale_factor = 2
resampled_data, resampled_meta = resample_raster(src_raster, upscale_factor)

def test_resample():
    """ This function compares only data. """
    assert resampled_meta['crs'] == src_raster.meta['crs']
    assert np.array_equal(src_width * upscale_factor, resampled_meta['width'])
    assert np.array_equal(src_height * upscale_factor, resampled_meta['height'])
    t = src_raster.transform
    assert resampled_meta['transform'] == Affine(t.a / upscale_factor, t.b, t.c,
                                                 t.d, t.e / upscale_factor, t.f)


