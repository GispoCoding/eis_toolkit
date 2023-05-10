import geopandas
from beartype import beartype

from eis_toolkit.exceptions import MatchingCrsException


@beartype
def reproject_vector(geodataframe: geopandas.GeoDataFrame, target_crs: int) -> geopandas.GeoDataFrame:
    """Reprojects vector data to match given coordinate reference system (EPSG).

    Args:
        geodataframe: The vector dataframe to be reprojected.
        target_crs: Target CRS as an EPSG code.

    Returns:
        Reprojected vector data.
    """

    if geodataframe.crs.to_epsg() == target_crs:
        raise MatchingCrsException("Vector data is already in the target CRS.")

    reprojected_gdf = geodataframe.to_crs("epsg:" + str(target_crs))
    return reprojected_gdf
