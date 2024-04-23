from numbers import Number

import geopandas as gpd
import numpy as np
from beartype import beartype
from beartype.typing import Union
from rasterio import profiles, transform

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidParameterValueException, NonMatchingCrsException
from eis_toolkit.utilities.checks.raster import check_raster_profile


@beartype
def _idw_interpolation(
    geodataframe: gpd.GeoDataFrame,
    target_column: str,
    raster_width: int,
    raster_height: int,
    raster_transform: transform.Affine,
    power: Number,
) -> np.ndarray:

    points = np.array(geodataframe.geometry.apply(lambda geom: (geom.x, geom.y)).tolist())
    values = geodataframe[target_column].values

    pixel_size_x = raster_transform.a
    pixel_size_y = abs(raster_transform.e)
    grid_x_min = raster_transform.xoff
    grid_x_max = grid_x_min + raster_width * pixel_size_x
    grid_y_max = raster_transform.yoff
    grid_y_min = grid_y_max - raster_height * pixel_size_y

    x = np.linspace(grid_x_min, grid_x_max, raster_width)
    y = np.linspace(grid_y_min, grid_y_max, raster_height)
    y = y[::-1].reshape(-1, 1)

    interpolated_values = _idw_core(points[:, 0], points[:, 1], values, x, y, power)
    interpolated_values = interpolated_values.reshape(raster_height, raster_width)

    return interpolated_values


#  Distance calculations
def _idw_core(x, y, z, xi, yi: np.ndarray, power: Number) -> np.ndarray:
    over = np.zeros((len(yi), len(xi)))
    under = np.zeros((len(yi), len(xi)))
    for n in range(len(x)):
        dist = np.hypot(xi - x[n], yi - y[n])
        # Add a small epsilon to avoid division by zero
        dist = np.where(dist == 0, 1e-12, dist)
        dist = dist**power

        over += z[n] / dist
        under += 1.0 / dist

    interpolated_values = over / under
    return interpolated_values


@beartype
def idw(
    geodataframe: gpd.GeoDataFrame,
    target_column: str,
    raster_profile: Union[profiles.Profile, dict],
    power: Number = 2,
) -> np.ndarray:
    """Calculate inverse distance weighted (IDW) interpolation.

    Args:
        geodataframe: The vector dataframe to be interpolated.
        target_column: The column name with values for each geometry.
        raster_profile: The raster profile used for output grid properties. Needs to include at least
            crs, transform, width and height.
        power: The value for determining the rate at which the weights decrease. As power increases,
            the weights for distant points decrease rapidly. Defaults to 2.

    Returns:
        Numpy array containing the interpolated values.

    Raises:
        EmptyDataFrameException: The input GeoDataFrame is empty.
        InvalidParameterValueException: Invalid resolution or target_column.
        NonMatchingCrsException: The input GeoDataFrame and raster profile have mismatching CRS.
    """
    if geodataframe.empty:
        raise EmptyDataFrameException("Expected geodataframe to contain geometries.")
    if raster_profile.get("crs") != geodataframe.crs:
        raise NonMatchingCrsException("Expected coordinate systems to match between raster and GeoDataFrame.")
    if target_column not in geodataframe.columns:
        raise InvalidParameterValueException(
            f"Expected target_column ({target_column}) to be contained in geodataframe columns."
        )
    check_raster_profile(raster_profile)

    raster_width = raster_profile.get("width")
    raster_height = raster_profile.get("height")
    raster_transform = raster_profile.get("transform")

    interpolated_values = _idw_interpolation(
        geodataframe, target_column, raster_width, raster_height, raster_transform, power
    )

    return interpolated_values
