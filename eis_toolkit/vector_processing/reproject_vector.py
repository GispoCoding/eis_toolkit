import geopandas

from eis_toolkit.exceptions import MatchingCrsException


def reproject_vector(  # type: ignore[no-any-unimported]
    geodataframe: geopandas.GeoDataFrame, target_EPSG: int
) -> geopandas.GeoDataFrame:
    """Reprojects vector data to match given coordinate system (EPSG).

    Args:
        geodataframe (geopandas.GeoDataFrame): The vector dataframe to be reprojected.
        target_EPSG (int): Target crs as EPSG code.

    Returns:
        reprojected_gdf (geopandas.GeoDataFrame): Reprojected vector data.
    """

    if geodataframe.crs.to_epsg() == target_EPSG:
        raise MatchingCrsException

    reprojected_gdf = geodataframe.to_crs("epsg:" + str(target_EPSG))
    return reprojected_gdf
