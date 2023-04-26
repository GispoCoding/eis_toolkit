import sys
from pathlib import Path

import geopandas
import pytest
from pyproj.exceptions import ProjError

from eis_toolkit.exceptions import MatchingCrsException
from eis_toolkit.vector_processing.reproject_vector import reproject_vector

parent_dir = Path(__file__).parent
vector_path = parent_dir.joinpath("data/remote/small_area.shp")
reference_solution_path = parent_dir.joinpath("data/remote/small_area_reprojected.shp")

geodataframe = geopandas.read_file(vector_path)
reference_geodataframe = geopandas.read_file(reference_solution_path)


@pytest.mark.xfail(
    sys.platform == "win32", reason="Fails due to internal pyproj transformation error on Windows.", raises=ProjError
)
def test_reproject_vector():
    """Test reproject vector functionality."""
    reprojected_geodataframe = reproject_vector(geodataframe, 4326)
    assert reprojected_geodataframe.crs == reference_geodataframe.crs
    assert all(reprojected_geodataframe == reference_geodataframe)


def test_same_crs():
    """Test that a crs match raises the correct exception."""
    with pytest.raises(MatchingCrsException):
        reproject_vector(geodataframe, 3067)
