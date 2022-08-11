import shapely
from eis_toolkit.exceptions import NotApplicableGeometryTypeException


def check_geometry_type(geometry: shapely.geometry, allowed_types: list):
    """Checks the type of a given geometry against a list of allowed types.

    Args:
        geometry: a shapely.geometry object.
        allowed_types: a list of allowed geometry types

    Raises:
        TODO
    """

    type = geometry.geom_type
    if type not in allowed_types:
        raise NotApplicableGeometryTypeException(
            f"{type}, should be one of: {allowed_types}"
        )


def check_geometry_types(geometries: list, allowed_types: list):
    [check_geometry_type(geometry, allowed_types) for geometry in geometries]
