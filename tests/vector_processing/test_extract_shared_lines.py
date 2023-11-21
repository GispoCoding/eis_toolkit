
import geopandas as gpd
import pytest
from shapely.geometry import Polygon
from shapely.geometry import MultiLineString
from shapely.geometry import LineString

from eis_toolkit.vector_processing.extract_shared_lines import extract_shared_lines
from eis_toolkit.exceptions import EmptyDataFrameException, InvalidParameterValueException

@pytest.fixture
def example_polygons():
    poly1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    poly2 = Polygon([(1, 0), (1, 1), (2, 1), (2, 0)])
    poly3 = Polygon([(2, 0), (2, 1), (3, 1), (3, 0)])

    data = {'geometry': [poly1, poly2, poly3]}
    gdf = gpd.GeoDataFrame(data)
    return gdf

def test_validated_extracted_shared_lines(example_polygons):
    result = extract_shared_lines(example_polygons)
    expected_lines = [
        LineString([(1, 1), (1, 0)]),
        LineString([(1, 0), (1, 1)]),
        LineString([(2, 0), (2, 1)]),
        LineString([(2, 1), (2, 0)])
    ]

    result = extract_shared_lines(example_polygons)
    
    assert len(result) == len(expected_lines) / 2, "Unexpected amount of lines"

    for line in result['geometry']:
        assert any(line.equals(expected_line) for expected_line in expected_lines), f"Line {line} was not found in expected lines"

@pytest.fixture
def example_not_enough_polygons():
    poly1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])

    data = {'geometry': [poly1]}
    gdf = gpd.GeoDataFrame(data)
    return gdf

@pytest.fixture
def example_empty_geodataframe():
    gdf = gpd.GeoDataFrame()
    return gdf

def test_empy_geodataframe(example_empty_geodataframe):
    with pytest.raises(EmptyDataFrameException):
        extract_shared_lines(example_empty_geodataframe)

def test_not_enough_polygons(example_not_enough_polygons):
    with pytest.raises(InvalidParameterValueException):
        extract_shared_lines(example_not_enough_polygons)