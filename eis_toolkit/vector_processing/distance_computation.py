from numbers import Number

import geopandas as gpd
import numpy as np
from beartype import beartype
from beartype.typing import Optional, Union
from rasterio import profiles, transform
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
