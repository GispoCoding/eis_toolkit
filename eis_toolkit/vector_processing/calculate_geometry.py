import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from beartype import beartype
from eis_toolkit.exceptions import EmptyDataFrameException

@beartype
def calculate_geometry(geodataframe: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculates the lenght or area of the given geometries.
    
    Args:
        geodataframe: Geometries to be calculated.

    Returns:
        calculated_gdf: Geometries and calculated values-

    Raises:
        EmptyDataFrameException if input geodataframe is empty.
    """
    if geodataframe.shape[0] == 0:
        raise EmptyDataFrameException("Expected geodataframe to contain geometries.")

    calculated_gdf = geodataframe.copy()

    calculated_gdf['calculated_value'] = None

    for index, row in calculated_gdf.iterrows():
        geometry_type = row['geometry'].geom_type

        if geometry_type == 'Point':
            calculated_gdf.at[index, 'calculated_value'] = 0
        elif geometry_type == 'LineString':
            calculated_gdf.at[index, 'calculated_value'] = row['geometry'].length
        elif geometry_type == 'Polygon':
            calculated_gdf.at[index, 'calculated_value'] = row['geometry'].area

    return calculated_gdf