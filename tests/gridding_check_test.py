import rasterio

from pathlib import Path
from eis_toolkit.raster_processing.gridding_check import gridding_check

parent_dir = Path(__file__).parent
snap_raster_path = parent_dir.joinpath("data/remote/snap_raster.tif")
small_raster_path = parent_dir.joinpath("data/remote/small_raster.tif")
small_raster_epsg4326_path = parent_dir.joinpath("data/remote/small_raster_EPSG4326.tif")
snap_raster_smaller_cells_path = parent_dir.joinpath("data/remote/snap_test_raster_smaller_cells.tif")
snap_raster_top_right_path = parent_dir.joinpath("data/remote/snap_test_raster_right_top.tif")
snap_raster_bottom_right_path = parent_dir.joinpath("data/remote/snap_test_raster_right_bottom.tif")
clipped_snap_raster_path = parent_dir.joinpath("data/remote/clipped_snap_raster.tif")

snap_raster = rasterio.open(snap_raster_path)
small_raster = rasterio.open(small_raster_path)
small_raster_epsg4326 = rasterio.open(small_raster_epsg4326_path)
snap_raster_smaller_cells = rasterio.open(snap_raster_smaller_cells_path)
snap_raster_top_right = rasterio.open(snap_raster_top_right_path)
snap_raster_bottom_right = rasterio.open(snap_raster_bottom_right_path)
clipped_snap_raster = rasterio.open(clipped_snap_raster_path)


def test_rasters_with_matching_gridding() -> None:
    """Check that gridding_check returns True when pixel alignment,
    crs and cell size matches, with same_extent set to false"""
    test = gridding_check([snap_raster, snap_raster, clipped_snap_raster])
    assert test is True


def test_rasters_with_matching_gridding_same_extent() -> None:
    """Check that gridding_check returns False when same_extent is set to true, while
    crs, cell size and pixel alignment matches, but bounds do not."""
    test = gridding_check([snap_raster, snap_raster, snap_raster, clipped_snap_raster], True)
    assert test is False


def test_crs_false() -> None:
    """Test that nonmatching crs returns false"""
    test = gridding_check([small_raster, small_raster_epsg4326])
    assert test is False


def test_crs_false_same_extent() -> None:
    """Test that nonmatching crs returns false with same_extent set to True."""
    test = gridding_check([small_raster, small_raster_epsg4326], True)
    assert test is False


def test_cell_size_false() -> None:
    """Test that nonmatching cell size returns False."""
    test = gridding_check([snap_raster, snap_raster_smaller_cells, snap_raster])
    assert test is False


def test_cell_size_false_same_extent() -> None:
    """Test that nonmatching cell size returns False with same_extent set to True."""
    test = gridding_check([snap_raster_smaller_cells, snap_raster], True)
    assert test is False


def test_alignment_false_same_extent() -> None:
    """Test that nonmatching pixelalignment returns False with same_extent set to True."""
    test = gridding_check([snap_raster, snap_raster_bottom_right, clipped_snap_raster], True)
    assert test is False


def test_alignment_false() -> None:
    """Test that matching pixelalignment returns True, with same_extent set to False."""
    test = gridding_check([snap_raster, snap_raster_bottom_right, clipped_snap_raster])
    assert test is True
