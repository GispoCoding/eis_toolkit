from numbers import Number

import geopandas as gpd
import numpy as np
import numpy.ma as ma
from beartype import beartype
from beartype.typing import Tuple
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

from eis_toolkit.exceptions import (
    EmptyDataFrameException,
    InvalidParameterValueException,
    NotApplicableGeometryTypeException,
)


def _kriging(
    data: gpd.GeoDataFrame,
    resolution: Tuple[Number, Number],
    extent: Tuple[Number, Number, Number, Number],
    variogram_model: str,
    method: str,
    drift_terms: list,
) -> Tuple[np.ndarray, dict]:

    coordinates = np.array(list(data.geometry.apply(lambda geom: [geom.x, geom.y, geom.z])))
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    z = coordinates[:, 2]

    grid_x = np.linspace(extent[0], extent[1], resolution[0])
    grid_y = np.linspace(extent[2], extent[3], resolution[1])

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
    resolution: Tuple[Number, Number],
    extent: Tuple[Number, Number, Number, Number],
    variogram_model: str = "linear",
    method: str = "ordinary",
    drift_terms: list = ["regional_linear"],
) -> Tuple[np.ndarray, dict]:
    """
    Perform Kriging interpolation on the input data.

    Args:
        data: GeoDataFrame containing the input data.
        resolution: Size of the output grid.
        extent: Limits of the output grid.
        variogram_model: Variogram model to be used. Defaults to 'linear'.
        method: Kriging method. Defaults to 'ordinary'.
        drift_terms: Drift terms used in universal kriging.

    Returns:
        Grid containing the interpolated values and metadata.

    Raises:
        EmptyDataFrameException: The input GeoDataFrame is empty.
        InvalidParameterValueException: The resolution is not greater than zero, variogram model is invalid,
                                        kriging method is invalid or any input drift term is invalid.
        NotApplicableGeometryTypeException: GeoDataFrame's geometry is missing z coordinates.
    """

    if data.empty:
        raise EmptyDataFrameException("The input GeoDataFrame is empty.")

    if resolution[0] <= 0 or resolution[1] <= 0:
        raise InvalidParameterValueException("The resolution must be greater than zero.")

    if False in set(data.geometry.has_z):
        raise NotApplicableGeometryTypeException("Data points must have z coordinates.")

    if variogram_model not in ("linear", "power", "gaussian", "spherical", "exponential", "hole-effect"):
        raise InvalidParameterValueException(
            "Variogram model must be 'linear', 'power', 'gaussian', 'spherical', 'exponential' or 'hole-effect'."
        )

    if method not in ("ordinary", "universal"):
        raise InvalidParameterValueException("Kriging method must be either 'ordinary' or 'universal'.")

    if any(
        term not in ("regional_linear", "point_log", "external_Z", "specified", "functional") for term in drift_terms
    ):
        raise InvalidParameterValueException(
            "Accepted drift terms are 'regional_linear', 'point_log', 'external_Z', 'specified' and 'functional'."
        )

    data_interpolated, out_meta = _kriging(data, resolution, extent, variogram_model, method, drift_terms)

    return data_interpolated, out_meta
