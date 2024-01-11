from beartype import beartype
from beartype.typing import Iterable
from rasterio.profiles import Profile


@beartype
def check_matching_crs(objects: Iterable) -> bool:
    """Check if every object in a list has a CRS, and that they match.

    Args:
        objects: A list of objects to check.

    Returns:
        True if everything matches, False if not.
    """
    epsg_list = []

    for object in objects:
        if not type(object) == Profile:
            if not object.crs:
                return False
            epsg = object.crs.to_epsg()
            epsg_list.append(epsg)
        else:
            epsg_list.append(object["crs"])

    if len(set(epsg_list)) != 1:
        return False

    return True
