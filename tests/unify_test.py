from pathlib import Path

import pytest
import rasterio

from rasterio.enums import Resampling

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.raster_processing.unifying import unify_rasters

parent_dir = Path(__file__).parent
base_raster_path = parent_dir.joinpath("data/remote/small_raster.tif")
base_raster = rasterio.open(base_raster_path)
raster_to_unify_path_1 = parent_dir.joinpath("data/remote/raster_to_unify.tif")
raster_to_unify_save_path_1 = parent_dir.joinpath("data/local/unified_res_1.tif")
raster_to_unify_path_2 = parent_dir.joinpath("data/remote/raster_to_unify_2.tif")
raster_to_unify_save_path_2 = parent_dir.joinpath("data/local/unified_res_2.tif")


def test_unify_rasters_case1():
    """Tests extract_window function in Case 1.
    
    The raster-to-unfiy here needs all 3 of reprojecting, resampling and snapping.
    The result pixel size is not exactly the same as the base rasters and the unified
    result does not look the best as it is nonsquare pixels projected from WGS 84"""
    raster_to_unify = rasterio.open(raster_to_unify_path_1)
    out_rasters = unify_rasters(base_raster, [raster_to_unify], Resampling.nearest)
    out_image, out_meta = out_rasters[1]

    assert len(out_rasters) == 2
    assert out_meta["crs"] == base_raster.crs
    assert abs(out_meta["transform"].a - base_raster.transform.a) < 0.1
    assert abs(out_meta["transform"].e - base_raster.transform.e) < 0.1

    with rasterio.open(raster_to_unify_save_path_1, 'w', **out_meta) as dst:
        dst.write(out_image)


def test_unify_rasters_case2():
    """Tests extract_window function in Case 2.
    
    The raster-to-unify here needs resampling and snapping."""
    
    raster_to_unify = rasterio.open(raster_to_unify_path_2)
    out_rasters = unify_rasters(base_raster, [raster_to_unify], Resampling.bilinear)
    out_image, out_meta = out_rasters[1]

    assert len(out_rasters) == 2
    assert out_meta["crs"] == base_raster.crs
    assert abs(out_meta["transform"].a - base_raster.transform.a) < 0.1
    assert abs(out_meta["transform"].e - base_raster.transform.e) < 0.1
    # TODO: test values and nodata

    with rasterio.open(raster_to_unify_save_path_2, 'w', **out_meta) as dst:
        dst.write(out_image)
    

def test_unify_rasters_empty_raster_list():
    """Test that empty raster list raises correct exception."""
    with pytest.raises(InvalidParameterValueException):
        _ = unify_rasters(base_raster, [])


def test_unify_rasters_wrong_type_in_raster_list():
    """Test that wrong type of items in raster list raises correct exception."""
    with pytest.raises(InvalidParameterValueException):
        _ = unify_rasters(base_raster, [1, 3, 5])
