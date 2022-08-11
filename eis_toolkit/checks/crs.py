from eis_toolkit.exceptions import NonMatchingCrsException


def check_crs_matches(objects: list):
    """TODO Checks if every object in a list has the same crs."""

    epsg_list = []
    for object in objects:
        if not object.crs:
            raise NonMatchingCrsException
        epsg = object.crs.to_epsg()
        epsg_list.append(epsg)
    if len(set(epsg_list)) != 1:
        raise NonMatchingCrsException
