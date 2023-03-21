from pathlib import Path

import rasterio

from eis_toolkit.raster_processing.gridding_check import gridding_check

# Test rasters.
parent_dir = Path(__file__).parent
snap_raster = rasterio.open(parent_dir.joinpath("data/remote/snap_raster.tif"))
small_raster = rasterio.open(parent_dir.joinpath("data/remote/small_raster.tif"))
small_raster_epsg4326 = rasterio.open(parent_dir.joinpath("data/remote/small_raster_EPSG4326.tif"))
snap_raster_smaller_cells = rasterio.open(parent_dir.joinpath("data/remote/snap_test_raster_smaller_cells.tif"))
snap_raster_top_right = rasterio.open(parent_dir.joinpath("data/remote/snap_test_raster_right_top.tif"))
snap_raster_bottom_right = rasterio.open(parent_dir.joinpath("data/remote/snap_test_raster_right_bottom.tif"))
clipped_snap_raster = rasterio.open(parent_dir.joinpath("data/remote/clipped_snap_raster.tif"))


def test_identical_rasters_same_extent() -> None:
    """Check that identical rasters return True."""
    test = gridding_check([snap_raster, snap_raster, snap_raster], True)
    assert test is True


def test_rasters_with_matching_gridding() -> None:
    """Check that matching gridding returns True."""
    test = gridding_check([snap_raster, snap_raster, clipped_snap_raster])
    assert test is True


def test_crs_false() -> None:
    """Test that nonmatching crs returns False."""
    test = gridding_check([small_raster, small_raster_epsg4326])
    assert test is False


def test_cell_size_false_same_extent() -> None:
    """Test that nonmatching cell size returns False with same_extent set to True."""
    test = gridding_check([snap_raster_smaller_cells, snap_raster], True)
    assert test is False


def test_alignment_false_same_extent() -> None:
    """Test that nonmatching pixel alignment returns False with same_extent set to True."""
    test = gridding_check([snap_raster, snap_raster, clipped_snap_raster], True)
    assert test is False


def test_alignment_false() -> None:
    """Test that matching pixel alignment returns True, with same_extent set to False."""
    test = gridding_check([snap_raster, snap_raster, clipped_snap_raster])
    assert test is True
