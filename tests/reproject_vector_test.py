from pathlib import Path

import geopandas
import pytest

from eis_toolkit.vector_processing.reproject_vector import reproject_vector
from eis_toolkit.exceptions import MatchingCrsException


parent_dir = Path(__file__).parent
vector_path = parent_dir.joinpath("data/remote/small_area.shp")
reference_solution_path = parent_dir.joinpath("data/remote/small_area_reprojected.shp")

geodataframe = geopandas.read_file(vector_path)
reference_geodataframe = geopandas.read_file(reference_solution_path)


def test_reproject_vector():
    """Test reproject vector functionality."""
    reprojected_geodataframe = reproject_vector(geodataframe, 4326)
    assert reprojected_geodataframe.crs == reference_geodataframe.crs
    assert all(reprojected_geodataframe == reference_geodataframe)


def test_same_crs():
    """Test that a crs match raises the correct exception."""
    with pytest.raises(MatchingCrsException):
        reproject_vector(geodataframe, 3067)