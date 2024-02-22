import geopandas as gpd
import numpy as np
from beartype import beartype
from beartype.typing import Union
from rasterio import profiles, transform
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidParameterValueException, NonMatchingCrsException


@beartype
def distance_computation(raster_profile: Union[profiles.Profile, dict], geometries: gpd.GeoDataFrame) -> np.ndarray:
    """Calculate distance from raster cell to nearest geometry.

    Args:
        raster_profile: The raster profile of the raster in which the distances
            to the nearest geometry are determined.
        geometries: The geometries to determine distance to.

    Returns:
        A 2D numpy array with the distances computed.

    """
    if raster_profile.get("crs") != geometries.crs:
        raise NonMatchingCrsException("Expected coordinate systems to match between raster and geometries. ")
    if geometries.shape[0] == 0:
        raise EmptyDataFrameException("Expected GeoDataFrame to not be empty.")

    raster_width = raster_profile.get("width")
    raster_height = raster_profile.get("height")

    if not isinstance(raster_width, int) or not isinstance(raster_height, int):
        raise InvalidParameterValueException(
            f"Expected raster_profile to contain integer width and height. {raster_profile}"
        )

    raster_transform = raster_profile.get("transform")

    if not isinstance(raster_transform, transform.Affine):
        raise InvalidParameterValueException(
            f"Expected raster_profile to contain an affine transformation. {raster_profile}"
        )

    return _distance_computation(
        raster_width=raster_width, raster_height=raster_height, raster_transform=raster_transform, geometries=geometries
    )


def _calculate_row_distances(
    row: int,
    cols: np.ndarray,
    raster_transform: transform.Affine,
    geometries_unary_union: Union[BaseGeometry, BaseMultipartGeometry],
) -> np.ndarray:
    # transform.xy accepts either cols or rows as an array. The other then has
    # to be an integer. The resulting x and y point coordinates are therefore
    # in a 1D array
    point_xs, point_ys = transform.xy(transform=raster_transform, cols=cols, rows=row)
    row_points = [Point(x, y) for x, y in zip(point_xs, point_ys)]
    row_distances = np.array([point.distance(geometries_unary_union) for point in row_points])
    return row_distances


def _distance_computation(
    raster_width: int, raster_height: int, raster_transform: transform.Affine, geometries: gpd.GeoDataFrame
) -> np.ndarray:

    cols = np.arange(raster_width)
    rows = np.arange(raster_height)

    geometries_unary_union = geometries.geometry.unary_union

    distance_matrix = np.array(
        [
            _calculate_row_distances(
                row=row, cols=cols, raster_transform=raster_transform, geometries_unary_union=geometries_unary_union
            )
            for row in rows
        ]
    )

    return distance_matrix
