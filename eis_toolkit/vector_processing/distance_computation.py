from numbers import Number

import geopandas as gpd
import numpy as np
from beartype import beartype
from beartype.typing import Optional, Union
from numba import njit, prange
from rasterio import profiles, transform

from eis_toolkit import exceptions
from eis_toolkit.utilities.checks.raster import check_raster_profile


@beartype
def distance_computation(
    geodataframe: gpd.GeoDataFrame, raster_profile: Union[profiles.Profile, dict], max_distance: Optional[Number] = None
) -> np.ndarray:
    """
    Calculate distance from each raster cell (centre) to the nearest input geometry.

    Pixels on top of input geometries are assigned distance of 0.

    Uses Numba to perform calculations quickly. The computation time increases (roughly)
    linearly with the amount of raster pixels defined by given `raster_profile`. Supports
    Polygon, MultiPolygon, LineString, MultiLineString, Point and MultiPoint geometries.

    Args:
        geodataframe: The GeoDataFrame with geometries to determine distance to.
        raster_profile: The raster profile of the raster in which the distances
            to the nearest geometry are determined.
        max_distance: The maximum distance in the output array. Pixels beyond this
            distance will be assigned `max_distance` value.

    Returns:
        A 2D numpy array with the distances computed.

    Raises:
        NonMatchingCrsException: The input raster profile and geodataframe have mismatching CRS.
        EmptyDataFrameException: The input geodataframe is empty.
        NumericValueSignException: Max distance is defined and is not a positive number.
    """
    if raster_profile.get("crs") != geodataframe.crs:
        raise exceptions.NonMatchingCrsException(
            "Expected coordinate systems to match between raster and GeoDataFrame."
        )
    if geodataframe.empty:
        raise exceptions.EmptyDataFrameException("Expected GeoDataFrame to not be empty.")
    if max_distance is not None and max_distance <= 0:
        raise exceptions.NumericValueSignException("Expected max distance to be a positive number.")

    check_raster_profile(raster_profile=raster_profile)

    raster_width = raster_profile.get("width")
    raster_height = raster_profile.get("height")
    raster_transform = raster_profile.get("transform")

    # Generate the grid of raster cell center points
    raster_points = _generate_raster_points(raster_width, raster_height, raster_transform)

    # Initialize lists needed for Numba-compatible calculations
    segment_coords = []  # These will also contain points coords, if present
    segment_indices = [0]  # Start index
    polygon_coords = []
    polygon_indices = [0]  # Start index

    for geometry in geodataframe.geometry:
        if geometry.geom_type == "Polygon":
            coords = list(geometry.exterior.coords)
            for x, y in coords:
                polygon_coords.extend([x, y])
            polygon_indices.append(len(polygon_coords) // 2)
            segments = [
                (coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1]) for i in range(len(coords) - 1)
            ]

        elif geometry.geom_type == "MultiPolygon":
            # For MultiPolygon, iterate over each polygon
            segments = []
            for poly in geometry.geoms:
                coords = list(poly.exterior.coords)
                for x, y in coords:
                    polygon_coords.extend([x, y])
                polygon_indices.append(len(polygon_coords) // 2)

                # Add polygon boundary as segments for distance calculations
                segments.extend(
                    [(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1]) for i in range(len(coords) - 1)]
                )

        elif geometry.geom_type == "LineString":
            coords = list(geometry.coords)
            segments = [
                (coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1]) for i in range(len(coords) - 1)
            ]

        elif geometry.geom_type == "MultiLineString":
            # For MultiLineString, iterate through each line string component
            segments = []
            for line in geometry.geoms:
                coords = list(line.coords)
                segments.extend(
                    [(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1]) for i in range(len(coords) - 1)]
                )

        elif geometry.geom_type == "Point":
            segments = [(geometry.x, geometry.y)]

        elif geometry.geom_type == "MultiPoint":
            # For MultiPoint, iterate over each point and add as individual (x, y) tuples
            segments = [(point.x, point.y) for point in geometry.geoms]

        else:
            raise exceptions.GeometryTypeException(f"Encountered unsupported geometry type: {geometry.geom_type}.")

        segment_coords.extend(segments)
        segment_indices.append(len(segment_coords))  # End index for this geometry's segments

    # Convert all lists to numpy arrays
    segment_coords = np.array(segment_coords, dtype=np.float64)
    segment_indices = np.array(segment_indices, dtype=np.int64)
    polygon_coords = np.array(polygon_coords, dtype=np.float64)
    polygon_indices = np.array(polygon_indices, dtype=np.int64)

    distance_matrix = _compute_distances_core(
        raster_points,
        segment_coords,
        segment_indices,
        polygon_coords,
        polygon_indices,
        raster_width,
        raster_height,
        max_distance,
    )

    return distance_matrix


@njit(parallel=True)
def _compute_distances_core(
    raster_points: np.ndarray,
    segment_coords: np.ndarray,
    segment_indices: np.ndarray,
    polygon_coords: np.ndarray,
    polygon_indices: np.ndarray,
    width: int,
    height: int,
    max_distance: Optional[Number],
) -> np.ndarray:
    distance_matrix = np.full((height, width), np.inf)
    for i in prange(len(raster_points)):
        px, py = raster_points[i]
        min_dist = np.inf

        # Check if the point is inside any polygon, if polygons are present
        if len(polygon_indices) > 1 and _point_in_polygon(px, py, polygon_coords, polygon_indices):
            min_dist = 0  # Set distance to zero if point is inside a polygon
        else:
            # Only calculate distance to segments if point is outside all polygons
            for j in range(len(segment_indices) - 1):
                for k in range(segment_indices[j], segment_indices[j + 1]):
                    # Case 1: Point
                    if len(segment_coords[k]) == 2:
                        x1, y1 = segment_coords[k]
                        dist = np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
                    # Case 2: Line segment
                    else:
                        x1, y1, x2, y2 = segment_coords[k]
                        dist = _point_to_segment_distance(px, py, x1, y1, x2, y2)
                    if dist < min_dist:
                        min_dist = dist

        # Apply max_distance threshold if specified
        if max_distance is not None:
            min_dist = min(min_dist, max_distance)

        # Update the distance matrix
        distance_matrix[i // width, i % width] = min_dist
    return distance_matrix


def _generate_raster_points(width: int, height: int, affine_transform: transform.Affine) -> np.ndarray:
    """Generate a full grid of points from the raster dimensions and affine transform."""
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = transform.xy(affine_transform, rows, cols)
    points = np.column_stack([np.array(xs).ravel(), np.array(ys).ravel()])
    return points


@njit
def _point_to_segment_distance(px: Number, py: Number, x1: Number, y1: Number, x2: Number, y2: Number) -> np.ndarray:
    """Calculate the minimum distance from a point to a line segment."""
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        # Segment is a point (Should not happen)
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    nearest_x, nearest_y = x1 + t * dx, y1 + t * dy
    return np.sqrt((px - nearest_x) ** 2 + (py - nearest_y) ** 2)


@njit
def _point_in_polygon(px: Number, py: Number, polygon_coords: np.ndarray, polygon_indices: np.ndarray) -> bool:
    """Determine if a point is inside any polygon using the ray-casting algorithm."""
    for p_start, p_end in zip(polygon_indices[:-1], polygon_indices[1:]):
        inside = False
        xints = 0.0
        n = p_end - p_start
        p1x, p1y = polygon_coords[2 * p_start], polygon_coords[2 * p_start + 1]
        for i in range(n + 1):
            p2x, p2y = polygon_coords[2 * (p_start + i % n)], polygon_coords[2 * (p_start + i % n) + 1]
            if py > min(p1y, p2y):
                if py <= max(p1y, p2y):
                    if px <= max(p1x, p2x):
                        if p1y != p2y:
                            xints = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or px <= xints:
                            inside = not inside
            p1x, p1y = p2x, p2y
        if inside:
            return True
    return False


# @beartype
# def distance_computation(
#     geodataframe: gpd.GeoDataFrame,
#     raster_profile: Union[profiles.Profile, dict],
#     max_distance: Optional[Number] = None
# ) -> np.ndarray:
#     """Calculate distance from raster cell to nearest geometry.

#     Args:
#         geodataframe: The GeoDataFrame with geometries to determine distance to.
#         raster_profile: The raster profile of the raster in which the distances
#             to the nearest geometry are determined.
#         max_distance: The maximum distance in the output array.

#     Returns:
#         A 2D numpy array with the distances computed.

#     Raises:
#         NonMatchingCrsException: The input raster profile and geodataframe have mismatching CRS.
#         EmptyDataFrameException: The input geodataframe is empty.
#         NumericValueSignException: Max distance is defined and is not a positive number.
#     """
#     if raster_profile.get("crs") != geodataframe.crs:
#         raise exceptions.NonMatchingCrsException(
#             "Expected coordinate systems to match between raster and GeoDataFrame."
#         )
#     if geodataframe.shape[0] == 0:
#         raise exceptions.EmptyDataFrameException("Expected GeoDataFrame to not be empty.")
#     if max_distance is not None and max_distance <= 0:
#         raise exceptions.NumericValueSignException("Expected max distance to be a positive number.")

#     check_raster_profile(raster_profile=raster_profile)

#     raster_width = raster_profile.get("width")
#     raster_height = raster_profile.get("height")
#     raster_transform = raster_profile.get("transform")

#     distance_matrix = _distance_computation(
#         raster_width=raster_width,
#         raster_height=raster_height,
#         raster_transform=raster_transform,
#         geodataframe=geodataframe,
#     )
#     if max_distance is not None:
#         distance_matrix[distance_matrix > max_distance] = max_distance

#     return distance_matrix


# def _calculate_row_distances(
#     row: int,
#     cols: np.ndarray,
#     raster_transform: transform.Affine,
#     geometries_unary_union: Union[BaseGeometry, BaseMultipartGeometry],
# ) -> np.ndarray:
#     row_distances = np.array(
#         [
#             point.distance(geometries_unary_union)
#             for point in row_points(row=row, cols=cols, raster_transform=raster_transform)
#         ]
#     )
#     return row_distances


# def _distance_computation(
#     raster_width: int, raster_height: int, raster_transform: transform.Affine, geodataframe: gpd.GeoDataFrame
# ) -> np.ndarray:

#     cols = np.arange(raster_width)
#     rows = np.arange(raster_height)

#     geometries_unary_union = geodataframe.geometry.unary_union

#     distance_matrix = np.array(
#         [
#             _calculate_row_distances(
#                 row=row, cols=cols, raster_transform=raster_transform, geometries_unary_union=geometries_unary_union
#             )
#             for row in rows
#         ]
#     )

#     return distance_matrix
