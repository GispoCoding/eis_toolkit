from numbers import Number
from typing import Tuple

import geopandas as gpd
import numpy as np
import numpy.ma as ma
from beartype import beartype
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidParameterValueException


def _kriging(data: gpd.GeoDataFrame, resolution: Tuple[Number, Number], limits: list, method: str) -> np.ndarray:

    coordinates = np.array(list(data.geometry.apply(lambda geom: [geom.x, geom.y, geom.z])))
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    z = coordinates[:, 2]

    grid_x = np.linspace(limits[0][0], limits[0][1], resolution[0])
    grid_y = np.linspace(limits[1][0], limits[1][1], resolution[1])

    if method == "universal_kriging":
        universal_kriging = UniversalKriging(x, y, z, variogram_model="linear", drift_terms=["regional_linear"])
        z_interpolated, sigma_squared = universal_kriging.execute("grid", grid_x, grid_y)

    else:
        ordinary_kriging = OrdinaryKriging(x, y, z, variogram_model="linear")
        z_interpolated, sigma_squared = ordinary_kriging.execute("grid", grid_x, grid_y)

    z_interpolated = ma.getdata(z_interpolated)

    return z_interpolated


@beartype
def kriging(
    data: gpd.GeoDataFrame, resolution: Tuple[Number, Number], limits: list, method: str = "ordinary_kriging"
) -> np.ndarray:
    """
    Perform Kriging interpolation on the input data.

    Args:
        data: GeoDataFrame containing the input data.
        resolution: Size of the output grid.
        limits: Limits of the output grid.
        method: Kriging algorithm. Defaults to Ordinary Kriging.

    Returns:
        z_interpolated: Grid containing the interpolated values.

    Raises:
        EmptyDataFrameException: The input GeoDataFrame is empty.
        InvalidParameterValueException: The resolution is not greater than zero.
    """

    if data.empty:
        raise EmptyDataFrameException("The input GeoDataFrame is empty.")

    if sum(resolution) <= 0:
        raise InvalidParameterValueException("The input value for resolution must be greater than zero.")

    data_interpolated = _kriging(data, resolution, limits, method)

    return data_interpolated
