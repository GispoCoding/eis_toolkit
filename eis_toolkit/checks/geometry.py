from typing import Iterable


def check_geometry_types(geometries: Iterable, allowed_types: list):
    """Checks all geometries in an iterable against a list of allowed types.

    Args:
        geometries (Iterable): for example a list of shapely.geometry objects
        or a geopandas.GeoSeries.
        allowed_types: a list of allowed geometry types.

    Returns:
        Bool: True if all geometries match, False if not
    """

    for geometry in geometries:
        if geometry.geom_type not in allowed_types:
            return False
    return True
