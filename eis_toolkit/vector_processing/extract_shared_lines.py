import geopandas as gpd
from beartype import beartype

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidParameterValueException


def _extract_shared_lines(polygons: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    shared_lines_list = []

    # Used to make sure that same polygon couples are not examined together multiple times
    reversed_couples = set()

    for i, poly1 in polygons.iterrows():
        for j, poly2 in polygons.iterrows():
            if i < j and (i, j) not in reversed_couples:
                reversed_couples.add((i, j))
                if i != j:
                    shared_line = poly1["geometry"].intersection(poly2["geometry"])
                    if not shared_line.is_empty and (
                        shared_line.geom_type == "MultiLineString" or shared_line.geom_type == "LineString"
                    ):
                        shared_lines_list.append(shared_line)

    shared_lines_gdf = gpd.GeoDataFrame(geometry=shared_lines_list)
    shared_lines_gdf["ID"] = shared_lines_gdf.reset_index().index

    return shared_lines_gdf


@beartype
def extract_shared_lines(polygons: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Extract shared lines/borders/edges between polygons.

    Args:
        polygons: The geodataframe that contains the polygon geometries to be examined
            for shared lines.

    Returns:
        Geodataframe containing the shared lines that were found between the polygons.

    Raises:
        EmptyDataFrameException if input geodataframe is empty.
        InvalidParameterValueException if input geodataframe doesn't contain at least 2 polygons.
    """
    if polygons.shape[0] == 0:
        raise EmptyDataFrameException("Geodataframe is empty.")

    if polygons.shape[0] < 2:
        raise InvalidParameterValueException("Expected GeoDataFrame to have at least 2 polygons.")

    shared_lines = _extract_shared_lines(polygons)

    return shared_lines
