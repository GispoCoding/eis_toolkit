import geopandas as gpd
from beartype import beartype

from eis_toolkit.exceptions import EmptyDataFrameException


@beartype
def calculate_geometry(geodataframe: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate the length or area of the given geometries.

    Args:
        geodataframe: Geometries to be calculated.

    Returns:
        calculated_gdf: Geometries and calculated values.

    Raises:
        EmptyDataFrameException if input geodataframe is empty.
    """
    if geodataframe.shape[0] == 0:
        raise EmptyDataFrameException("Expected geodataframe to contain geometries.")

    calculated_gdf = geodataframe.copy()
    calculated_gdf["value"] = calculated_gdf.apply(lambda row: _calculate_value(row), axis=1)

    return calculated_gdf


def _calculate_value(row):
    geometry_type = row["geometry"].geom_type

    if geometry_type in ["Point", "MultiPoint"]:
        return 0
    elif geometry_type in ["LineString", "MultiLineString"]:
        return row["geometry"].length
    elif geometry_type in ["Polygon", "MultiPolygon"]:
        return row["geometry"].area
