from beartype import beartype
from beartype.typing import Iterable


@beartype
def check_geometry_types(geometries: Iterable, allowed_types: Iterable) -> bool:
    """Check all geometries in an iterable against a list of allowed types.

    Args:
        geometries: Geometries to check (for example shapely.geometry objects or a geopandas.GeoSeries).
        allowed_types: Allowed geometry types.

    Returns:
        True if all geometries match, False if not.
    """
    for geometry in geometries:
        if geometry.geom_type not in allowed_types:
            return False

    return True
