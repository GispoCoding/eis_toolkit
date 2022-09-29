from pathlib import Path

import numpy as np
import pytest
import rasterio

from eis_toolkit.exceptions import MatchingCrsException
from eis_toolkit.raster_processing.reprojecting import reproject_raster

parent_dir = Path(__file__).parent
raster_path = parent_dir.joinpath("data/remote/small_raster.tif")
reference_solution_path = parent_dir.joinpath("data/remote/small_raster_EPSG4326.tif")

src_raster = rasterio.open(raster_path)
reprojected_data, reprojected_meta = reproject_raster(src_raster, 4326)

reference_raster = rasterio.open(reference_solution_path)
reference_data = reference_raster.read()
reference_meta = reference_raster.meta


def test_reproject_data():
    """This function compares only data."""
    assert np.array_equal(reprojected_data, reference_data)


def test_reproject_meta():
    """This function compares only meta data."""
    assert reprojected_meta["count"] == reference_meta["count"]
    assert reprojected_meta["crs"] == reference_meta["crs"]
    assert reprojected_meta["driver"] == reference_meta["driver"]
    assert reprojected_meta["dtype"] == reference_meta["dtype"]
    assert reprojected_meta["height"] == reference_meta["height"]
    assert reprojected_meta["width"] == reference_meta["width"]
    assert abs(reprojected_meta['transform'][0] - reference_meta['transform'][0]) < 0.00000001
    assert abs(reprojected_meta['transform'][1] - reference_meta['transform'][1]) < 0.00000001
    assert abs(reprojected_meta['transform'][2] - reference_meta['transform'][2]) < 0.00000001
    assert abs(reprojected_meta['transform'][3] - reference_meta['transform'][3]) < 0.00000001
    assert abs(reprojected_meta['transform'][4] - reference_meta['transform'][4]) < 0.00000001
    assert abs(reprojected_meta['transform'][5] - reference_meta['transform'][5]) < 0.00000001


def test_same_crs():
    """Test that a crs match raises the correct exception."""
    with pytest.raises(MatchingCrsException):
        with rasterio.open(raster_path) as raster:
            reproject_raster(raster, int(raster.crs.to_string()[5:]))
