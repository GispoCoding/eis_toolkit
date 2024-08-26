from pathlib import Path

import numpy as np
import pytest
import rasterio

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.raster_processing.unifying import unify_raster_grids
from tests.raster_processing.masking_test import small_raster_clipped_path as base_raster_path_3

test_dir = Path(__file__).parent.parent

base_raster_path_1 = test_dir.joinpath("data/remote/small_raster.tif")
base_raster_path_2 = test_dir.joinpath("data/remote/smaller_raster.tif")

raster_to_unify_path_1 = test_dir.joinpath("data/remote/unifying/raster_to_unify_1.tif")
raster_to_unify_path_2 = test_dir.joinpath("data/remote/unifying/raster_to_unify_2.tif")

raster_to_unify_save_path_1 = test_dir.joinpath("data/local/results/unify_test_result_1.tif")
raster_to_unify_save_path_2 = test_dir.joinpath("data/local/results/unify_test_result_2.tif")


def test_unify_raster_grids():
    """
    Tests unify raster grids without masking.

    The raster-to-unify differes from the base raster in...
    - crs
    - pixel size
    - alignment
    """

    with rasterio.open(raster_to_unify_path_1) as raster_to_unify:
        with rasterio.open(base_raster_path_1) as base_raster:
            out_rasters = unify_raster_grids(base_raster, [raster_to_unify], "nearest", masking=None)
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


def test_unify_raster_grids_extent():
    """
    Tests unify grids with extent masking.

    The raster-to-unify differs from the base raster in...
    - pixel size
    - alignment
    - bounds
    """

    with rasterio.open(raster_to_unify_path_2) as raster_to_unify:
        with rasterio.open(base_raster_path_2) as base_raster:
            out_rasters = unify_raster_grids(base_raster, [raster_to_unify], "bilinear", masking="extents")
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


def test_unify_raster_grids_full_masking():
    """
    Tests unify raster grids with full masking (extent and nodata).

    The raster-to-unify differs from the base raster in...
    - pixel size
    - alignment
    - bounds
    """

    with rasterio.open(raster_to_unify_path_1) as raster_to_unify:
        with rasterio.open(base_raster_path_3) as base_raster:
            out_rasters = unify_raster_grids(base_raster, [raster_to_unify], "bilinear", masking="full")
            out_image, out_profile = out_rasters[1]

            assert len(out_rasters) == 2
            assert out_profile["crs"] == base_raster.crs
            # Check pixel size
            assert out_profile["transform"].a == base_raster.transform.a
            assert out_profile["transform"].e == base_raster.transform.e
            # Check grid alignment and bounds
            assert out_profile["transform"].c == base_raster.transform.c
            assert out_profile["transform"].f == base_raster.transform.f
            assert out_profile["width"] == base_raster.width
            assert out_profile["height"] == base_raster.height
            # Check nodata locations
            np.testing.assert_array_equal(
                base_raster.read(1) == base_raster.nodata, out_image[0] == out_profile["nodata"]
            )

    with rasterio.open(raster_to_unify_save_path_2, "w", **out_profile) as dst:
        dst.write(out_image)


def test_unify_raster_grids_empty_raster_list():
    """Test that empty raster list raises correct exception."""
    with pytest.raises(InvalidParameterValueException):
        with rasterio.open(base_raster_path_1) as base_raster:
            _ = unify_raster_grids(base_raster, [])
