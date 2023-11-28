import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidParameterValueException
from eis_toolkit.vector_processing.extract_shared_lines import extract_shared_lines


@pytest.fixture
def example_polygons():
    """Test data, 3 squares next to each other sharing 1 line with another square."""
    poly1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    poly2 = Polygon([(1, 0), (1, 1), (2, 1), (2, 0)])
    poly3 = Polygon([(2, 0), (2, 1), (3, 1), (3, 0)])

    data = {"geometry": [poly1, poly2, poly3]}
    gdf = gpd.GeoDataFrame(data)
    return gdf


def test_validated_extracted_shared_lines(example_polygons):
    """Test function works with valid data."""
    result = extract_shared_lines(example_polygons)
    expected_lines = [LineString([(1, 1), (1, 0)]), LineString([(2, 1), (2, 0)])]

    result = extract_shared_lines(example_polygons)

    assert len(result) == len(expected_lines), "Unexpected amount of lines"

    assert result["geometry"].equals(gpd.GeoSeries(expected_lines))


@pytest.fixture
def example_not_enough_polygons():
    """Test data for a test where there is not enough polygons in the geodataframe."""
    poly1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])

    data = {"geometry": [poly1]}
    gdf = gpd.GeoDataFrame(data)
    return gdf


@pytest.fixture
def example_empty_geodataframe():
    """Test data, empty geodataframe."""
    gdf = gpd.GeoDataFrame()
    return gdf


def test_empy_geodataframe(example_empty_geodataframe):
    """Test empty gdf raises an exception."""
    with pytest.raises(EmptyDataFrameException):
        extract_shared_lines(example_empty_geodataframe)


def test_not_enough_polygons(example_not_enough_polygons):
    """Test not enough polygons raises an exception."""
    with pytest.raises(InvalidParameterValueException):
        extract_shared_lines(example_not_enough_polygons)
