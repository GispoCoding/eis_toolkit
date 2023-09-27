from pathlib import Path

import numpy as np
import pytest
import rasterio

from eis_toolkit.exceptions import NonMatchingCrsException
from eis_toolkit.raster_processing.reprojecting import reproject_raster
from eis_toolkit.raster_processing.resampling import resample
from eis_toolkit.raster_processing.snapping import snap_with_raster

test_dir = Path(__file__).parent.parent

snap_raster_path = test_dir.joinpath("data/remote/snapping/snap_raster.tif")
case1_raster_path = test_dir.joinpath("data/remote/snapping/snap_test_raster_right_top.tif")
case2_raster_path = test_dir.joinpath("data/remote/snapping/snap_test_raster_right_bottom.tif")
case3_raster_path = test_dir.joinpath("data/remote/snapping/snap_test_raster_smaller_cells.tif")
case4_raster_path = test_dir.joinpath("data/remote/snapping/snap_test_raster_outofbounds.tif")
case5_raster_path = test_dir.joinpath("data/remote/small_raster_multiband.tif")
nonsquare_raster_path = test_dir.joinpath("data/remote/snapping/snap_test_raster_nonsquare.tif")

# Save some test rasters to local
wrong_crs_path = test_dir.joinpath("data/local/results/snap_test_wrong_crs.tif")
small_snap_raster_path = test_dir.joinpath("data/local/results/snap_test_small_snap_raster.tif")


def test_snap_case1_to_right_top():
    """Test snap functionality case 1. Same pixel sizes, raster values should snap towards right top."""
    raster = rasterio.open(case1_raster_path)
    snap_raster = rasterio.open(snap_raster_path)
    out_image, out_meta = snap_with_raster(raster, snap_raster)

    assert out_meta["height"] == raster.height + 1
    assert out_meta["width"] == raster.width + 1
    assert out_meta["transform"].c == raster.meta["transform"].c - 1.5
    assert out_meta["transform"].f == raster.meta["transform"].f + 0.5
    assert np.array_equal(out_image[:, :-1, 1:], raster.read())


def test_snap_case2_to_right_bottom():
    """Test snap functionality case 2. Same pixel sizes, raster values should snap towards right bottom."""
    raster = rasterio.open(case2_raster_path)
    snap_raster = rasterio.open(snap_raster_path)
    out_image, out_meta = snap_with_raster(raster, snap_raster)

    assert out_meta["height"] == raster.height + 1
    assert out_meta["width"] == raster.width + 1
    assert out_meta["transform"].c == raster.meta["transform"].c - 1.5
    assert out_meta["transform"].f == raster.meta["transform"].f + 1.5
    assert np.array_equal(out_image[:, 1:, 1:], raster.read())


def test_snap_case3_to_left_top_smaller_pixels():
    """Test snap functionality case 3. Raster with smaller pixels, raster should snap towards left top."""
    raster = rasterio.open(case3_raster_path)
    snap_raster = rasterio.open(snap_raster_path)
    out_image, out_meta = snap_with_raster(raster, snap_raster)

    assert out_meta["height"] == raster.height + 2
    assert out_meta["width"] == raster.width + 2
    assert out_meta["transform"].c == raster.meta["transform"].c - 1.5
    assert out_meta["transform"].f == raster.meta["transform"].f + 1.7
    assert np.array_equal(out_image[:, 1:-1, 1:-1], raster.read())


def test_snap_case4_to_left_bottom_outside_snap_raster():
    """Test snap functionality case 4. Raster snap corner outside snap raster, should snap towards left bottom."""
    raster = rasterio.open(case4_raster_path)
    snap_raster = rasterio.open(snap_raster_path)
    out_image, out_meta = snap_with_raster(raster, snap_raster)

    assert out_meta["height"] == raster.height + 1
    assert out_meta["width"] == raster.width + 1
    assert out_meta["transform"].c == raster.meta["transform"].c - 0.5
    assert out_meta["transform"].f == raster.meta["transform"].f + 1.5
    assert np.array_equal(out_image[:, 1:, :-1], raster.read())


def test_snap_case5_to_right_top_multiband():
    """Test snap functionality case 5. Raster is multiband, should snap towards right top."""
    raster = rasterio.open(case5_raster_path)
    snap_raster = rasterio.open(snap_raster_path)
    out_image, out_meta = snap_with_raster(raster, snap_raster)

    assert out_meta["height"] == raster.height + 1
    assert out_meta["width"] == raster.width + 1
    assert out_meta["transform"].c == raster.meta["transform"].c - 1.5
    assert out_meta["transform"].f == raster.meta["transform"].f + 0.5
    assert np.array_equal(out_image[:, :-1, 1:], raster.read())


def test_snap_case6_small_snap_raster_pixel_size():
    """
    Test snap functionality case 6.

    Snap raster has smaller pixel size than to-be-snapped raster. Should snap towards left bottom.
    """
    # Resample snap raster to smaller pixel size than raster and write to local
    snap_raster = rasterio.open(snap_raster_path)
    out_image, out_meta = resample(snap_raster, snap_raster.meta["transform"][0] / 3)
    with rasterio.open(small_snap_raster_path, "w", **out_meta) as dest:
        dest.write(out_image)

    raster = rasterio.open(case1_raster_path)
    snap_raster = rasterio.open(small_snap_raster_path)
    out_image, out_meta = snap_with_raster(raster, snap_raster)

    assert out_meta["height"] == raster.height + 1
    assert out_meta["width"] == raster.width + 1
    assert round(out_meta["transform"].c - raster.meta["transform"].c, 4) == -0.1667
    assert round(out_meta["transform"].f - raster.meta["transform"].f, 4) == 1.8333
    assert np.array_equal(out_image[:, 1:, :-1], raster.read())


def test_snap_different_crs():
    """Test that a crs mismatch raises the correct exception."""
    with pytest.raises(NonMatchingCrsException):
        # Reproject to different crs and write reprojected raster to local
        raster = rasterio.open(case1_raster_path)
        out_image, out_meta = reproject_raster(raster, 4326)
        with rasterio.open(wrong_crs_path, "w", **out_meta) as dest:
            dest.write(out_image)

        raster = rasterio.open(wrong_crs_path)
        snap_raster = rasterio.open(snap_raster_path)
        _, _ = snap_with_raster(raster, snap_raster)
