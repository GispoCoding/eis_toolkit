from pathlib import Path

import geopandas
import pytest

from eis_toolkit.vector_processing.reproject_vector import reproject_vector
from eis_toolkit.exceptions import MatchingCrsException


parent_dir = Path(__file__).parent
vector_path = parent_dir.joinpath("data/remote/small_area.shp")
geodataframe = geopandas.read_file(vector_path)


def test_reproject_vector():
    """Test reproject vector functionality."""
    reprojected_geodataframe = reproject_vector(geodataframe, 4326)
    assert reprojected_geodataframe.crs.to_epsg() == 4326


def test_same_crs():
    """Test that a crs match raises the correct exception."""
    with pytest.raises(MatchingCrsException):
        # small_area.shp is in epsg:3067
        reproject_vector(geodataframe, 3067)