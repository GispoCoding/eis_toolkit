from numbers import Number

import geopandas as gpd
import numpy as np
from beartype import beartype
from beartype.typing import Optional, Tuple, Union

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidParameterValueException


@beartype
def _simple_idw(
    geodataframe: gpd.GeoDataFrame,
    target_column: str,
    resolution: Tuple[Number, Number],
    power: Number,
    extent: Union[Tuple[Number, Number, Number, Number], None],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

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

    num_points_x = int((x_max - x_min) / resolution_x)
    num_points_y = int((y_max - y_min) / resolution_y)

    x = np.linspace(x_min, x_max, num_points_x)
    y = np.linspace(y_min, y_max, num_points_y)

    xi, yi = np.meshgrid(x, y)
    xi = xi.flatten()
    # Reverse the order of y-values
    yi = yi[::-1].flatten()

    origin_x, origin_y = 0, 0
    dist_from_origin = np.hypot(points[:, 0] - origin_x, points[:, 1] - origin_y)
    sorted_indices = np.argsort(dist_from_origin)
    sorted_points = points[sorted_indices]
    sorted_values = values[sorted_indices]

    interpolated_values = _idw(sorted_points[:, 0], sorted_points[:, 1], sorted_values, xi, yi, power)
    interpolated_values = interpolated_values.reshape(num_points_y, num_points_x)

    return x, y, interpolated_values


#  Distance calculations
def _distance_matrix(x0, y0, x1, y1: np.ndarray) -> np.ndarray:
    d0 = np.subtract.outer(x0, x1)
    d1 = np.subtract.outer(y0, y1)
    return np.hypot(d0, d1)


def _idw(x, y, z, xi, yi: np.ndarray, power: Number) -> np.ndarray:
    dist = _distance_matrix(x, y, xi, yi)
    # Add a small epsilon to avoid division by zero
    dist = np.where(dist == 0, 1e-12, dist)
    weights = 1.0 / (dist**power)
    weights /= weights.sum(axis=0)
    interpolated_values = np.dot(weights.T, z)

    return interpolated_values


def simple_idw(
    geodataframe: gpd.GeoDataFrame,
    target_column: str,
    resolution: Tuple[Number, Number],
    extent: Optional[Tuple[Number, Number, Number, Number]] = None,
    power: Optional[Number] = 2.0,
) -> Tuple[float, float, dict]:
    """Calculate simple inverse distance weighted (IDW) interpolation.

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

    Raises:
        EmptyDataFrameException: The input GeoDataFrame is empty.
        InvalidParameterValueException: Invalid resolution or target_column.
    """

    if geodataframe.shape[0] == 0:
        raise EmptyDataFrameException("Expected geodataframe to contain geometries.")

    if target_column not in geodataframe.columns:
        raise InvalidParameterValueException(
            f"Expected target_column ({target_column}) to be contained in geodataframe columns."
        )

    if resolution[0] <= 0 or resolution[1] <= 0:
        raise InvalidParameterValueException("Expected height and width greater than zero.")

    x, y, interpolated_values = _simple_idw(geodataframe, target_column, resolution, power, extent)
    return x, y, interpolated_values
