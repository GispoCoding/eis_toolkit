from pathlib import Path

import pytest
import rasterio
from rasterio.enums import Resampling

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.raster_processing.unifying import unify_raster_grids

test_dir = Path(__file__).parent.parent

base_raster_path_1 = test_dir.joinpath("data/remote/small_raster.tif")
base_raster_path_2 = test_dir.joinpath("data/remote/smaller_raster.tif")

raster_to_unify_path_1 = test_dir.joinpath("data/remote/unifying/raster_to_unify_1.tif")
raster_to_unify_path_2 = test_dir.joinpath("data/remote/unifying/raster_to_unify_2.tif")

raster_to_unify_save_path_1 = test_dir.joinpath("data/local/results/unify_test_result_1.tif")
raster_to_unify_save_path_2 = test_dir.joinpath("data/local/results/unify_test_result_2.tif")


def test_unify_raster_grids_case1():
    """
    Tests unify function for case 1.

    Nearest resampling with same_extent set to False (no clipping).

    The raster-to-unify differes from the base raster in...
    - crs
    - pixel size
    - alignment
    """

    with rasterio.open(raster_to_unify_path_1) as raster_to_unify:
        with rasterio.open(base_raster_path_1) as base_raster:
            out_rasters = unify_raster_grids(base_raster, [raster_to_unify], Resampling.nearest, False)
            out_image, out_meta = out_rasters[1]

            assert len(out_rasters) == 2
            assert out_meta["crs"] == base_raster.crs
            # Check pixel size
            assert out_meta["transform"].a == base_raster.transform.a
            assert out_meta["transform"].e == base_raster.transform.e
            # Check grid alignment
            assert abs(out_meta["transform"].c - base_raster.transform.c) % base_raster.transform.a == 0
            assert abs(out_meta["transform"].f - base_raster.transform.f) % abs(base_raster.transform.e) == 0

    with rasterio.open(raster_to_unify_save_path_1, "w", **out_meta) as dst:
        dst.write(out_image)


def test_unify_raster_grids_case2():
    """
    Tests unify function for case 2.

    Bilinear resampling with same_extent set to True (clipping).

    The raster-to-unify differs from the base raster in...
    - pixel size
    - alignment
    - bounds
    """

    with rasterio.open(raster_to_unify_path_2) as raster_to_unify:
        with rasterio.open(base_raster_path_2) as base_raster:
            out_rasters = unify_raster_grids(base_raster, [raster_to_unify], Resampling.bilinear, True)
            out_image, out_meta = out_rasters[1]

            assert len(out_rasters) == 2
            assert out_meta["crs"] == base_raster.crs
            # Check pixel size
            assert out_meta["transform"].a == base_raster.transform.a
            assert out_meta["transform"].e == base_raster.transform.e
            # Check grid alignment and bounds
            assert out_meta["transform"].c == base_raster.transform.c
            assert out_meta["transform"].f == base_raster.transform.f
            assert out_meta["width"] == base_raster.width
            assert out_meta["height"] == base_raster.height

    with rasterio.open(raster_to_unify_save_path_2, "w", **out_meta) as dst:
        dst.write(out_image)


def test_unify_raster_grids_empty_raster_list():
    """Test that empty raster list raises correct exception."""
    with pytest.raises(InvalidParameterValueException):
        with rasterio.open(base_raster_path_1) as base_raster:
            _ = unify_raster_grids(base_raster, [])
