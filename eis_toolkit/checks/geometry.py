from typing import Iterable


def correct_geometry_types(geometries: Iterable, allowed: list):
    """Checks all geometries in an iterable against a list of allowed types.

    Args:
        geometries (Iterable): for example a list of shapely.geometry objects
        or a geopandas.GeoSeries.
        allowed_types: a list of allowed geometry types.

    Raises:
        TODO
    """

    for geometry in geometries:
        if geometry.geom_type not in allowed:
            return False
    return True
