import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

from eis_toolkit.exceptions import EmptyDataFrameException
from eis_toolkit.vector_processing.calculate_geometry import calculate_geometry


@pytest.fixture
def example_geometries():
    """Test data, Point, LineString and Polygon."""
    point = Point(10, 10)
    line = LineString([(0, 0), (0, 150)])
    polygon = Polygon([(0, 0), (10, 0), (10, 25), (0, 25)])

    data = {"geometry": [point, line, polygon]}

    return gpd.GeoDataFrame(data, geometry="geometry")


def test_validated_calculate_geometries(example_geometries):
    """Test function works with valid data."""
    result = calculate_geometry(example_geometries)

    expected_values = np.array([0, 150, 250])
    calculated_values = result["value"].values

    assert np.array_equal(calculated_values, expected_values), "Expected and calculated values do not match"


@pytest.fixture
def empty_gdf():
    """Test data empty gdf."""
    gdf = gpd.GeoDataFrame()
    return gdf


def test_empty_geodataframe(empty_gdf):
    """Test empty gdf raises an exception."""
    with pytest.raises(EmptyDataFrameException):
        calculate_geometry(empty_gdf)
