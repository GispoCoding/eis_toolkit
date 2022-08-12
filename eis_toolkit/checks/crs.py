def matching_crs(objects: list):
    """Checks if every object in a list has a crs, and that they match."""

    epsg_list = []
    for object in objects:
        if not object.crs:
            return False
        epsg = object.crs.to_epsg()
        epsg_list.append(epsg)
    if len(set(epsg_list)) != 1:
        return False
    return True
