from numbers import Number

import geopandas as gpd
import numpy as np
from beartype import beartype
from beartype.typing import Optional, Union
from numba import njit, prange
from rasterio import profiles, transform
from scipy.spatial import cKDTree
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry

from eis_toolkit.exceptions import EmptyDataFrameException, NonMatchingCrsException, NumericValueSignException
from eis_toolkit.utilities.checks.raster import check_raster_profile
from eis_toolkit.utilities.miscellaneous import row_points


@beartype
def distance_computation(
    geodataframe: gpd.GeoDataFrame, raster_profile: Union[profiles.Profile, dict], max_distance: Optional[Number] = None
) -> np.ndarray:
    """Calculate distance from raster cell to nearest geometry.

    Args:
        geodataframe: The GeoDataFrame with geometries to determine distance to.
        raster_profile: The raster profile of the raster in which the distances
            to the nearest geometry are determined.
        max_distance: The maximum distance in the output array.

    Returns:
        A 2D numpy array with the distances computed.

    Raises:
        NonMatchingCrsException: The input raster profile and geodataframe have mismatching CRS.
        EmptyDataFrameException: The input geodataframe is empty.
    """
    if raster_profile.get("crs") != geodataframe.crs:
        raise NonMatchingCrsException("Expected coordinate systems to match between raster and GeoDataFrame.")
    if geodataframe.shape[0] == 0:
        raise EmptyDataFrameException("Expected GeoDataFrame to not be empty.")
    if max_distance is not None and max_distance <= 0:
        raise NumericValueSignException("Expected max distance to be a positive number.")

    check_raster_profile(raster_profile=raster_profile)

    raster_width = raster_profile.get("width")
    raster_height = raster_profile.get("height")
    raster_transform = raster_profile.get("transform")

    distance_matrix = _distance_computation(
        raster_width=raster_width,
        raster_height=raster_height,
        raster_transform=raster_transform,
        geodataframe=geodataframe,
    )
    if max_distance is not None:
        distance_matrix[distance_matrix > max_distance] = max_distance

    return distance_matrix


def _calculate_row_distances(
    row: int,
    cols: np.ndarray,
    raster_transform: transform.Affine,
    geometries_unary_union: Union[BaseGeometry, BaseMultipartGeometry],
) -> np.ndarray:
    row_distances = np.array(
        [
            point.distance(geometries_unary_union)
            for point in row_points(row=row, cols=cols, raster_transform=raster_transform)
        ]
    )
    return row_distances


def _distance_computation(
    raster_width: int, raster_height: int, raster_transform: transform.Affine, geodataframe: gpd.GeoDataFrame
) -> np.ndarray:

    cols = np.arange(raster_width)
    rows = np.arange(raster_height)

    geometries_unary_union = geodataframe.geometry.unary_union

    distance_matrix = np.array(
        [
            _calculate_row_distances(
                row=row, cols=cols, raster_transform=raster_transform, geometries_unary_union=geometries_unary_union
            )
            for row in rows
        ]
    )

    return distance_matrix


@beartype
def distance_computation_optimized(
    geodataframe: gpd.GeoDataFrame, raster_profile: Union[profiles.Profile, dict], max_distance: Optional[Number] = None
) -> np.ndarray:
    """Calculate distance from raster cell to nearest geometry.

    Args:
        geodataframe: The GeoDataFrame with geometries to determine distance to.
        raster_profile: The raster profile of the raster in which the distances
            to the nearest geometry are determined.
        max_distance: The maximum distance in the output array.

    Returns:
        A 2D numpy array with the distances computed.

    Raises:
        NonMatchingCrsException: The input raster profile and geodataframe have mismatching CRS.
        EmptyDataFrameException: The input geodataframe is empty.
    """
    if raster_profile.get("crs") != geodataframe.crs:
        raise NonMatchingCrsException("Expected coordinate systems to match between raster and GeoDataFrame.")
    if geodataframe.shape[0] == 0:
        raise EmptyDataFrameException("Expected GeoDataFrame to not be empty.")
    if max_distance is not None and max_distance <= 0:
        raise NumericValueSignException("Expected max distance to be a positive number.")

    check_raster_profile(raster_profile=raster_profile)

    raster_width = raster_profile.get("width")
    raster_height = raster_profile.get("height")
    raster_transform = raster_profile.get("transform")

    # Get geometry centroids for cKDTree
    centroids = np.array([geom.centroid.coords[0] for geom in geodataframe.geometry])

    # Build a spatial index
    tree = cKDTree(centroids)

    # Generate the grid of points representing raster cells
    grid_points = _generate_raster_points(raster_width, raster_height, raster_transform)

    # Query nearest points using cKDTree (this step is outside of Numba)
    _, indices = tree.query(grid_points)

    # Compute the actual distances (using Numba for performance)
    distance_matrix = _distance_computation_optimized(grid_points, centroids[indices], raster_width, raster_height)

    if max_distance is not None:
        distance_matrix[distance_matrix > max_distance] = max_distance

    return distance_matrix


def _generate_raster_points(width: int, height: int, affine_transform: transform.Affine) -> np.ndarray:
    """Generate a full grid of points from the raster dimensions and affine transform."""
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = transform.xy(affine_transform, rows, cols)
    points = np.column_stack([np.array(xs).ravel(), np.array(ys).ravel()])
    return points


@njit(parallel=True)
def _distance_computation_optimized(
    points: np.ndarray, nearest_centroids: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Compute Euclidean distances between points and their nearest centroids using Numba."""
    distances = np.empty(points.shape[0])

    for i in prange(points.shape[0]):
        point = points[i]
        centroid = nearest_centroids[i]
        distances[i] = np.sqrt((point[0] - centroid[0]) ** 2 + (point[1] - centroid[1]) ** 2)

    return distances.reshape((height, width))
