import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon

from eis_toolkit.exceptions import EmptyDataFrameException
from eis_toolkit.vector_processing.calculate_geometry import calculate_geometry


@pytest.fixture
def example_geometries():
    """Test data, Point, LineString and Polygon."""
    point = Point(10, 10)
    line = LineString([(0, 0), (0, 150)])
    polygon = Polygon([(0, 0), (10, 0), (10, 25), (0, 25)])

    return gpd.GeoDataFrame({"geometry": [point, line, polygon]})


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


@pytest.fixture
def example_geometries_multi():
    """Test data, MultiPoint, MultiLineString, MultiPolygon."""

    multi_point = MultiPoint([(15, 15), (20, 20)])
    multi_line = MultiLineString([[(0, 0), (0, 10)], [(0, 10), (10, 10)]])
    multi_polygon = MultiPolygon(
        [Polygon([(0, 0), (10, 0), (10, 5), (0, 5)]), Polygon([(0, 5), (10, 5), (10, 10), (0, 10)])]
    )

    data = {"geometry": [multi_point, multi_line, multi_polygon]}

    return gpd.GeoDataFrame(data, geometry="geometry")


def test_validated_calculate_geometries_multi(example_geometries_multi):
    """Test function works with valid data."""
    result = calculate_geometry(example_geometries_multi)

    expected_values = np.array([0, 20, 100])
    calculated_values = result["value"].values

    assert np.array_equal(calculated_values, expected_values), "Expected and calculated values do not match"
