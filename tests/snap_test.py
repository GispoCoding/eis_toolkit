import pytest
import rasterio
import numpy as np
from pathlib import Path

from eis_toolkit.raster_processing.resampling import resample
from eis_toolkit.raster_processing.reprojecting import reproject_raster
from eis_toolkit.raster_processing.snapping import snap_with_raster
from eis_toolkit.exceptions import (
    InvalidPixelSizeException,
    NonMatchingCrsException,
    NonSquarePixelSizeException,
    OutOfBoundsException,
)

parent_dir = Path(__file__).parent

snap_raster_path = parent_dir.joinpath("data/local/snap_raster.tif")
case1_raster_path = parent_dir.joinpath("data/local/snap_test_raster_right_top.tif")
case2_raster_path = parent_dir.joinpath("data/local/snap_test_raster_right_bottom.tif")
case3_raster_path = parent_dir.joinpath("data/local/snap_test_raster_smaller_cells.tif")
out_of_bounds_raster_path = parent_dir.joinpath("data/local/snap_test_raster_outofbounds.tif")
nonsquare_raster_path = parent_dir.joinpath("data/local/snap_test_raster_nonsquare.tif")

# Save some test rasters to local
wrong_crs_path = parent_dir.joinpath("data/local/snap_test_wrong_crs.tif")
small_snap_raster_path = parent_dir.joinpath("data/local/snap_test_small_snap_raster.tif")


def test_snap_case1():
    """Test snap functionality case 1. Same pixel sizes, raster values should snap towards right top."""
    raster = rasterio.open(case1_raster_path)
    snap_raster = rasterio.open(snap_raster_path)
    out_image, out_meta = snap_with_raster(raster, snap_raster)
    
    assert out_meta['height'] == raster.height + 1
    assert out_meta['width'] == raster.width + 1
    assert out_meta['transform'].c == snap_raster.meta['transform'].c + 2
    assert out_meta['transform'].f == snap_raster.meta['transform'].f
    assert np.array_equal(out_image[:, :-1, 1:], raster.read())


def test_snap_case2():
    """Test snap functionality case 2. Same pixel sizes, raster values should snap towards right bottom."""
    raster = rasterio.open(case2_raster_path)
    snap_raster = rasterio.open(snap_raster_path)
    out_image, out_meta = snap_with_raster(raster, snap_raster)

    assert out_meta['height'] == raster.height + 1
    assert out_meta['width'] == raster.width + 1
    assert out_meta['transform'].c == snap_raster.meta['transform'].c + 2
    assert out_meta['transform'].f == snap_raster.meta['transform'].f
    assert np.array_equal(out_image[:, 1:, 1:], raster.read())


def test_snap_case3():
    """Test snap functionality case 3. Raster with smaller pixels, raster should snap towards left bottom."""
    raster = rasterio.open(case3_raster_path)
    snap_raster = rasterio.open(snap_raster_path)
    out_image, out_meta = snap_with_raster(raster, snap_raster)

    assert out_meta['height'] == raster.height + 2
    assert out_meta['width'] == raster.width + 2
    assert out_meta['transform'].c == snap_raster.meta['transform'].c + 2
    assert out_meta['transform'].f == snap_raster.meta['transform'].f + 1.2
    assert np.array_equal(out_image[:, 1:-1, 1:-1], raster.read())


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


def test_snap_small_snap_pixel_size():
    """Test that too small snap raster pixel size raises the correct exception."""
    with pytest.raises(InvalidPixelSizeException):
        # Resample snap raster to smaller pixel size than raster and write to local
        snap_raster = rasterio.open(snap_raster_path)
        out_image, out_meta = resample(snap_raster, 3)
        with rasterio.open(small_snap_raster_path, "w", **out_meta) as dest:
            dest.write(out_image)
    
        raster = rasterio.open(case1_raster_path)
        snap_raster = rasterio.open(small_snap_raster_path)
        _, _ = snap_with_raster(raster, snap_raster)


def test_snap_nonsquare_pixel():
    """Test that nonsquare pixel raises the correct exception."""
    with pytest.raises(NonSquarePixelSizeException):
        raster = rasterio.open(nonsquare_raster_path)
        snap_raster = rasterio.open(snap_raster_path)
        _, _ = snap_with_raster(raster, snap_raster)


def test_snap_out_of_bounds():
    """Test that left-bottom corner of raster being outside snap raster raises the correct exception."""
    with pytest.raises(OutOfBoundsException):
        raster = rasterio.open(out_of_bounds_raster_path)
        snap_raster = rasterio.open(snap_raster_path)
        _, _ = snap_with_raster(raster, snap_raster)
