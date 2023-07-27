from numbers import Number

import geopandas as gpd
import numpy as np
import numpy.ma as ma
from beartype import beartype
from beartype.typing import Tuple
from pykrige.ok import OrdinaryKriging

from eis_toolkit.exceptions import (
    EmptyDataFrameException,
    InvalidParameterValueException,
    NotApplicableGeometryTypeException,
)


def _kriging(
    data: gpd.GeoDataFrame, resolution: Tuple[Number, Number], extent: Tuple[Number, Number, Number, Number]
) -> Tuple[np.ndarray, dict]:

    coordinates = np.array(list(data.geometry.apply(lambda geom: [geom.x, geom.y, geom.z])))
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    z = coordinates[:, 2]

    grid_x = np.linspace(extent[0], extent[1], resolution[0])
    grid_y = np.linspace(extent[2], extent[3], resolution[1])

    ordinary_kriging = OrdinaryKriging(x, y, z, variogram_model="linear")
    z_interpolated, _ = ordinary_kriging.execute("grid", grid_x, grid_y)

    z_interpolated = ma.getdata(z_interpolated)

    out_meta = {"crs": data.crs, "width": len(grid_x), "height": len(grid_y)}

    return z_interpolated, out_meta


@beartype
def kriging(
    data: gpd.GeoDataFrame, resolution: Tuple[Number, Number], extent: Tuple[Number, Number, Number, Number]
) -> Tuple[np.ndarray, dict]:
    """
    Perform Kriging interpolation on the input data.

    Args:
        data: GeoDataFrame containing the input data.
        resolution: Size of the output grid.
        extent: Limits of the output grid.

    Returns:
        Grid containing the interpolated values and metadata.

    Raises:
        EmptyDataFrameException: The input GeoDataFrame is empty.
        InvalidParameterValueException: The resolution is not greater than zero.
        NotApplicableGeometryTypeException: GeoDataFrame's geometry is missing z coordinates.
    """

    if data.empty:
        raise EmptyDataFrameException("The input GeoDataFrame is empty.")

    if resolution[0] <= 0 or resolution[1] <= 0:
        raise InvalidParameterValueException("The resolution must be greater than zero.")

    if False in set(data.geometry.has_z):
        raise NotApplicableGeometryTypeException("Data points must have z coordinates.")

    data_interpolated, out_meta = _kriging(data, resolution, extent)

    return data_interpolated, out_meta
