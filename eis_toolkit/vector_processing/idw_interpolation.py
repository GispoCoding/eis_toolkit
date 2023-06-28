from numbers import Number
from beartype import beartype
from beartype.typing import Optional, Tuple
import numpy as np
from shapely.geometry import Point
import geopandas as gpd

from eis_toolkit import exceptions


@beartype
def _idw_interpolation(
    geodataframe: gpd.GeoDataFrame,
    target_column: str,
    resolution: Tuple[Number, Number],
    extent: Optional[Tuple[float, float, float, float]] = None,
    power: Optional[int] = 2
) -> Tuple[float, float, np.ndarray]:

    """Simple inverse distance weighted (IDW) interpolation.

    Args:
        geodataframe: The vector dataframe to be interpolated.
        target_column: The column name with values for each geometry.
        resolution: The resolution i.e. cell size of the output raster.
        extent: The extent of the output raster.
            If None, calculate extent from the input vector data.
        power: The value for determining the rate at which the weights decrease.
            As power increases, the weights for distant points decrease rapidly.
            Defaults to 2.

    Returns:
        Rasterized vector data and metadata.
    """

    if geodataframe.empty:
        # Empty GeoDataFrame
        raise ValueError("Expected geodataframe to contain geometries.")

    if target_column not in geodataframe.columns:
        raise ValueError(f"Expected target_column ({target_column}) to be contained in geodataframe columns.")

    points = np.array(geodataframe.geometry.apply(lambda geom: (geom.x, geom.y)).tolist())
    values = geodataframe[target_column].values

    if extent is None:
        x_min = geodataframe.geometry.total_bounds[0]
        x_max = geodataframe.geometry.total_bounds[2]
        y_min = geodataframe.geometry.total_bounds[1]
        y_max = geodataframe.geometry.total_bounds[3]
    else:
        x_min, y_min, x_max, y_max = extent

    resolution_x, resolution_y = resolution

    num_points_x = int((x_max - x_min) / resolution_x) + 1
    num_points_y = int((y_max - y_min) / resolution_y) + 1

    x = np.linspace(x_min, x_max, num_points_x)
    y = np.linspace(y_min, y_max, num_points_y)

    xi, yi = np.meshgrid(x, y)
    xi = xi.flatten()
    yi = yi[::-1].flatten()  # Reverse the order of y-values

    origin_x, origin_y = 0, 0
    dist_from_origin = np.hypot(points[:, 0] - origin_x, points[:, 1] - origin_y)
    sorted_indices = np.argsort(dist_from_origin)
    sorted_points = points[sorted_indices]
    sorted_values = values[sorted_indices]

    interpolated_values = _simple_idw(sorted_points[:, 0], sorted_points[:, 1], sorted_values, xi, yi, power)
    interpolated_values = interpolated_values.reshape(num_points_y, num_points_x)

    return x, y, interpolated_values


def _distance_matrix(x0, y0, x1, y1):
    d0 = np.subtract.outer(x0, x1)
    d1 = np.subtract.outer(y0, y1)
    return np.hypot(d0, d1)


def _simple_idw(x, y, z, xi, yi, power=2):

    dist = _distance_matrix(x, y, xi, yi)

    # Add a small epsilon to avoid division by zero
    dist = np.where(dist == 0, 1e-12, dist)

    # In IDW, weights are 1 / (distance ^ power)
    weights = 1.0 / (dist**power)

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    return np.dot(weights.T, z)


def idw_interpolation(
    geodataframe: gpd.GeoDataFrame,
    target_column: str,
    resolution: Tuple[Number, Number],
    extent: Optional[Tuple[float, float, float, float]] = None,
    power: Optional[int] = 2
) -> Tuple[float, float, dict]:

    """Simple inverse distance weighted (IDW) interpolation.

    Args:
        geodataframe: The vector dataframe to be interpolated.
        target_column: The column name with values for each geometry.
        resolution: The resolution i.e. cell size of the output raster.
        extent: The extent of the output raster.
            If None, calculate extent from the input vector data.
        power: The value for determining the rate at which the weights decrease.
            As power increases, the weights for distant points decrease rapidly.
            Defaults to 2.

    Returns:
        Rasterized vector data and metadata.
    """
    x, y, interpolated_values = _idw_interpolation(
        geodataframe,
        target_column,
        resolution,
        extent,
        power
    )
    return x, y, interpolated_values
