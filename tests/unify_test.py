from pathlib import Path

import pytest
import rasterio

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.raster_processing.unifying import unify_rasters

parent_dir = Path(__file__).parent
base_raster_path = parent_dir.joinpath("data/remote/small_raster.tif")
base_raster = rasterio.open(base_raster_path)
raster_to_unify_path = parent_dir.joinpath("data/remote/raster_to_unify.tif")
raster_to_unify = rasterio.open(raster_to_unify_path)


def test_unify_rasters_case1():
    """Tests extract_window function in Case1."""
    out_rasters = unify_rasters(base_raster, [raster_to_unify])
    out_image, out_meta = out_rasters[0]

    # a = pixel size in x direction
    # e = pixel size in y direction
    # c = corner x coordinate
    # f = corner y coordinate

    assert len(out_rasters) == 1
    assert out_meta["crs"] == base_raster.crs
    assert out_meta["transform"].a == base_raster.transform.a
    assert out_meta["transform"].e == base_raster.transform.e
    assert out_meta["transform"].c % out_meta["transform"].a == base_raster.transform.c % base_raster.transform.a
    assert out_meta["transform"].f % out_meta["transform"].e == base_raster.transform.f % base_raster.transform.e
    # TODO: test values and nodata
    

def test_unify_rasters_empty_raster_list():
    """Test that empty raster list raises correct exception."""
    with pytest.raises(InvalidParameterValueException):
        _ = unify_rasters(base_raster, [])


def test_unify_rasters_wrong_type_in_raster_list():
    """Test that wrong type of items in raster list raises correct exception."""
    with pytest.raises(InvalidParameterValueException):
        _ = unify_rasters(base_raster, [1, 3, 5])
