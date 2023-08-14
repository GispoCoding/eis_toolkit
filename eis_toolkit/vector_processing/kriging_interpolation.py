from numbers import Number

import geopandas as gpd
import numpy as np
import numpy.ma as ma
from beartype import beartype
from beartype.typing import Literal, Optional, Tuple
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidParameterValueException


def _kriging(
    data: gpd.GeoDataFrame,
    target_column: str,
    resolution: Tuple[Number, Number],
    extent: Optional[Tuple[Number, Number, Number, Number]],
    variogram_model: Literal,
    method: Literal,
    drift_terms: list,
) -> Tuple[np.ndarray, dict]:

    points = np.array(list(data.geometry.apply(lambda geom: [geom.x, geom.y])))
    x = points[:, 0]
    y = points[:, 1]
    z = data[target_column].values

    if extent is None:
        grid_x_min = data.geometry.total_bounds[0]
        grid_x_max = data.geometry.total_bounds[2]
        grid_y_min = data.geometry.total_bounds[1]
        grid_y_max = data.geometry.total_bounds[3]

    else:
        grid_x_min, grid_x_max, grid_y_min, grid_y_max = extent

    grid_x = np.linspace(grid_x_min, grid_x_max, resolution[0])
    grid_y = np.linspace(grid_y_min, grid_y_max, resolution[1])

    if method == "universal":
        universal_kriging = UniversalKriging(x, y, z, variogram_model=variogram_model, drift_terms=drift_terms)
        z_interpolated, _ = universal_kriging.execute("grid", grid_x, grid_y)

    if method == "ordinary":
        ordinary_kriging = OrdinaryKriging(x, y, z, variogram_model=variogram_model)
        z_interpolated, _ = ordinary_kriging.execute("grid", grid_x, grid_y)

    z_interpolated = ma.getdata(z_interpolated)

    out_meta = {"crs": data.crs, "width": len(grid_x), "height": len(grid_y)}

    return z_interpolated, out_meta


@beartype
def kriging(
    data: gpd.GeoDataFrame,
    target_column: str,
    resolution: Tuple[Number, Number],
    extent: Optional[Tuple[Number, Number, Number, Number]] = None,
    variogram_model: Literal["linear", "power", "gaussian", "spherical", "exponential", "hole-effect"] = "linear",
    method: Literal["ordinary", "universal"] = "ordinary",
    drift_terms: list = ["regional_linear"],
) -> Tuple[np.ndarray, dict]:
    """
    Perform Kriging interpolation on the input data.

    Args:
        data: GeoDataFrame containing the input data.
        target_column: The column name with values for each geometry.
        resolution: Size of the output grid.
        extent: The extent of the output grid.
            If None, calculate extent from the input vector data.
        variogram_model: Variogram model to be used. Defaults to 'linear'.
        method: Kriging method. Defaults to 'ordinary'.
        drift_terms: Drift terms used in universal kriging.

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

    if any(
        term not in ("regional_linear", "point_log", "external_Z", "specified", "functional") for term in drift_terms
    ):
        raise InvalidParameterValueException(
            "Accepted drift terms are 'regional_linear', 'point_log', 'external_Z', 'specified' and 'functional'."
        )

    data_interpolated, out_meta = _kriging(
        data, target_column, resolution, extent, variogram_model, method, drift_terms
    )

    return data_interpolated, out_meta
