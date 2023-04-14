from functools import partial

import geopandas as gpd
import numpy as np
from rasterio import profiles, transform
from shapely.geometry import Point

from eis_toolkit import exceptions


def _smallest_distance_to_point(point: Point, geometries: gpd.GeoDataFrame) -> float:
    """Calculate smalled distance from geometries to point."""
    # This could be sped up by using a spatial index to first choose only
    # geometries that are close to the point
    distances_to_point = geometries.distance(point)
    smallest_distance = distances_to_point.min()
    return smallest_distance


def distance_computation(raster_profile: profiles.Profile, geometries: gpd.GeoDataFrame) -> np.ndarray:
    """Calculate distance from raster cell to nearest geometry.

    Args:
        raster_profile: The raster profile of the raster in which the distances
            to the nearest geometry are determined.
        geometries: The geometries to determine distance to.

    Returns:
        A 2D numpy array with the distances computed.

    """
    if raster_profile.get("crs") != geometries.crs:
        raise exceptions.NonMatchingCrsException("Expected coordinate systems to match between raster and geometries. ")
    if geometries.shape[0] == 0:
        raise exceptions.EmptyDataFrameException("Expected GeoDataFrame to not be empty.")

    raster_width = raster_profile.get("width")
    raster_height = raster_profile.get("height")

    if not isinstance(raster_width, int) or not isinstance(raster_height, int):
        raise exceptions.InvalidParameterValueException(
            f"Expected raster_profile to contain integer width and height. {raster_profile}"
        )

    raster_transform = raster_profile.get("transform")

    if not isinstance(raster_transform, transform.Affine):
        raise exceptions.InvalidParameterValueException(
            f"Expected raster_profile to contain an affine transformation. {raster_profile}"
        )

    return _distance_computation(
        raster_width=raster_width, raster_height=raster_height, raster_transform=raster_transform, geometries=geometries
    )


def _calculate_row_distances(
    row: int, cols: np.ndarray, raster_transform: transform.Affine, geometries: gpd.GeoDataFrame
) -> np.ndarray:
    # transform.xy accepts either cols or rows as an array. The other then has
    # to be an integer. The resulting x and y point coordinates are therefore
    # in a 1D array
    point_xs, point_ys = transform.xy(transform=raster_transform, cols=cols, rows=row)
    row_points = [Point(x, y) for x, y in zip(point_xs, point_ys)]
    row_distances = np.array([_smallest_distance_to_point(point=point, geometries=geometries) for point in row_points])
    return row_distances


def _distance_computation(
    raster_width: int, raster_height: int, raster_transform: transform.Affine, geometries: gpd.GeoDataFrame
) -> np.ndarray:

    cols = np.arange(raster_width)
    rows = np.arange(raster_height)

    # Use partial to create a function with added static arguments
    _calculate_row_distances_for_geometries = partial(
        _calculate_row_distances, cols=cols, raster_transform=raster_transform, geometries=geometries
    )
    distance_matrix = np.array([_calculate_row_distances_for_geometries(row=row) for row in rows])

    return distance_matrix
