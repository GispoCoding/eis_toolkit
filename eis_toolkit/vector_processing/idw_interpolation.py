from numbers import Number

import geopandas as gpd
import numpy as np
from beartype import beartype
from beartype.typing import Optional, Tuple
from rasterio import transform

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidParameterValueException


@beartype
def _idw_interpolation(
    geodataframe: gpd.GeoDataFrame,
    target_column: str,
    resolution: Tuple[Number, Number],
    power: Number,
    extent: Optional[Tuple[Number, Number, Number, Number]],
) -> Tuple[np.ndarray, dict]:

    points = np.array(geodataframe.geometry.apply(lambda geom: (geom.x, geom.y)).tolist())
    values = geodataframe[target_column].values

    if extent is None:
        x_min, y_min, x_max, y_max = geodataframe.geometry.total_bounds
    else:
        x_min, x_max, y_min, y_max = extent

    resolution_x, resolution_y = resolution

    num_points_x = int((x_max - x_min) / resolution_x)
    num_points_y = int((y_max - y_min) / resolution_y)

    x = np.linspace(x_min, x_max, num_points_x)
    y = np.linspace(y_min, y_max, num_points_y)
    y = y[::-1].reshape(-1,1)

    interpolated_values = _idw_core(points[:, 0], points[:, 1], values, x, y, power)
    interpolated_values = interpolated_values.reshape(num_points_y, num_points_x)

    out_meta = {
        "crs": geodataframe.crs,
        "width": len(x),
        "height": len(y),
        "transform": transform.from_bounds(x_min, y_min, x_max, y_max, len(x), len(y)),
    }

    return interpolated_values, out_meta


#  Distance calculations
def _idw_core(x, y, z, xi, yi: np.ndarray, power: Number) -> np.ndarray:
    over = np.zeros( (len(yi), len(xi)) )
    under = np.zeros( (len(yi), len(xi)) )
    for n in range(len(x)):
        dist = np.hypot(xi -x[n], yi -y[n])
        # Add a small epsilon to avoid division by zero
        dist = np.where(dist == 0, 1e-12, dist)
        dist = dist ** power

        over += (z[n] / dist)
        under += 1.0 / dist

    interpolated_values = over / under
    return interpolated_values


@beartype
def idw(
    geodataframe: gpd.GeoDataFrame,
    target_column: str,
    resolution: Tuple[Number, Number],
    extent: Optional[Tuple[Number, Number, Number, Number]] = None,
    power: Number = 2,
) -> Tuple[np.ndarray, dict]:
    """Calculate inverse distance weighted (IDW) interpolation.

    Args:
        geodataframe: The vector dataframe to be interpolated.
        target_column: The column name with values for each geometry.
        resolution: The resolution i.e. cell size of the output raster as (pixel_size_x, pixel_size_y).
        extent: The extent of the output raster as (x_min, x_max, y_min, y_max).
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

    interpolated_values, out_meta = _idw_interpolation(geodataframe, target_column, resolution, power, extent)

    return interpolated_values, out_meta
