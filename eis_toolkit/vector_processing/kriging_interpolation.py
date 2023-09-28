from numbers import Number

import geopandas as gpd
import numpy as np
from beartype import beartype
from beartype.typing import Literal, Optional, Tuple
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from rasterio import transform

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidParameterValueException


def _kriging(
    data: gpd.GeoDataFrame,
    target_column: str,
    resolution: Tuple[Number, Number],
    extent: Optional[Tuple[Number, Number, Number, Number]],
    variogram_model: Literal,
    coordinates_type: Literal,
    method: Literal,
) -> Tuple[np.ndarray, dict]:

    x = data.geometry.x
    y = data.geometry.y
    z = data[target_column].values

    if extent is None:
        grid_x_min = data.geometry.total_bounds[0]
        grid_x_max = data.geometry.total_bounds[2]
        grid_y_min = data.geometry.total_bounds[1]
        grid_y_max = data.geometry.total_bounds[3]

    else:
        grid_x_min, grid_x_max, grid_y_min, grid_y_max = extent

    grid_x = np.arange(grid_x_min, grid_x_max + resolution[0], resolution[0])
    grid_y = np.arange(grid_y_min, grid_y_max + resolution[1], resolution[1])

    if method == "universal":
        universal_kriging = UniversalKriging(x, y, z, variogram_model=variogram_model, drift_terms=["regional_linear"])
        z_interpolated, _ = universal_kriging.execute("grid", grid_x, grid_y)

    if method == "ordinary":
        ordinary_kriging = OrdinaryKriging(x, y, z, variogram_model=variogram_model, coordinates_type=coordinates_type)
        z_interpolated, _ = ordinary_kriging.execute("grid", grid_x, grid_y)

    out_meta = {
        "crs": data.crs,
        "width": len(grid_x),
        "height": len(grid_y),
        "transform": transform.from_bounds(grid_x_min, grid_y_min, grid_x_max, grid_y_max, len(grid_x), len(grid_y)),
    }

    return z_interpolated, out_meta


@beartype
def kriging(
    data: gpd.GeoDataFrame,
    target_column: str,
    resolution: Tuple[Number, Number],
    extent: Optional[Tuple[Number, Number, Number, Number]] = None,
    variogram_model: Literal["linear", "power", "gaussian", "spherical", "exponential"] = "linear",
    coordinates_type: Literal["euclidean", "geographic"] = "geographic",
    method: Literal["ordinary", "universal"] = "ordinary",
) -> Tuple[np.ndarray, dict]:
    """
    Perform Kriging interpolation on the input data.

    Args:
        data: GeoDataFrame containing the input data.
        target_column: The column name with values for each geometry.
        resolution: The resolution i.e. cell size of the output raster as (pixel_size_x, pixel_size_y).
        extent: The extent of the output raster as (x_min, x_max, y_min, y_max).
            If None, calculate extent from the input vector data.
        variogram_model: Variogram model to be used.
            Either 'linear', 'power', 'gaussian', 'spherical' or 'exponential'. Defaults to 'linear'.
        coordinates_type: Determines are coordinates on a plane ('euclidean') or a sphere ('geographic').
            Used only in ordinary kriging. Defaults to 'geographic'.
        method: Ordinary or universal kriging. Defaults to 'ordinary'.

    Returns:
        Grid containing the interpolated values and metadata.

    Raises:
        EmptyDataFrameException: The input GeoDataFrame is empty.
        InvalidParameterValueException: Target column name is invalid or resolution is not greater than zero.
    """

    if data.empty:
        raise EmptyDataFrameException("The input GeoDataFrame is empty.")

    if target_column not in data.columns:
        raise InvalidParameterValueException(
            f"Expected target_column ({target_column}) to be contained in geodataframe columns."
        )

    if resolution[0] <= 0 or resolution[1] <= 0:
        raise InvalidParameterValueException("The resolution must be greater than zero.")

    data_interpolated, out_meta = _kriging(
        data, target_column, resolution, extent, variogram_model, coordinates_type, method
    )

    return data_interpolated, out_meta
