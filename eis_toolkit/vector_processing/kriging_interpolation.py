import geopandas as gpd
import numpy as np
from beartype import beartype
from beartype.typing import Literal, Union
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from rasterio import profiles, transform

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidParameterValueException, NonMatchingCrsException
from eis_toolkit.utilities.checks.raster import check_raster_profile


def _kriging(
    data: gpd.GeoDataFrame,
    target_column: str,
    raster_width: int,
    raster_height: int,
    raster_transform: transform.Affine,
    variogram_model: Literal["linear", "power", "gaussian", "spherical", "exponential"],
    coordinates_type: Literal["euclidean", "geographic"],
    method: Literal["ordinary", "universal"],
) -> np.ndarray:

    x = data.geometry.x
    y = data.geometry.y
    z = data[target_column].values

    pixel_size_x = raster_transform.a
    pixel_size_y = abs(raster_transform.e)
    grid_x_min = raster_transform.xoff
    grid_x_max = grid_x_min + raster_width * pixel_size_x
    grid_y_max = raster_transform.yoff
    grid_y_min = grid_y_max - raster_height * pixel_size_y

    grid_x = np.arange(grid_x_min, grid_x_max, pixel_size_x)
    grid_y = np.arange(grid_y_min, grid_y_max, pixel_size_y)

    if method == "universal":
        kriging_method = UniversalKriging(x, y, z, variogram_model=variogram_model, drift_terms=["regional_linear"])
    elif method == "ordinary":
        kriging_method = OrdinaryKriging(x, y, z, variogram_model=variogram_model, coordinates_type=coordinates_type)
    z_interpolated, _ = kriging_method.execute("grid", grid_x, grid_y)

    return z_interpolated


@beartype
def kriging(
    geodataframe: gpd.GeoDataFrame,
    target_column: str,
    raster_profile: Union[profiles.Profile, dict],
    variogram_model: Literal["linear", "power", "gaussian", "spherical", "exponential"] = "linear",
    coordinates_type: Literal["euclidean", "geographic"] = "geographic",
    method: Literal["ordinary", "universal"] = "ordinary",
) -> np.ndarray:
    """
    Perform Kriging interpolation on the input data.

    Args:
        geodataframe: GeoDataFrame containing the input data.
        target_column: The column name with values for each geometry.
        raster_profile: The raster profile used for output grid properties. Needs to include at least
            crs, transform, width and height.
        variogram_model: Variogram model to be used. Either 'linear', 'power', 'gaussian', 'spherical'
            or 'exponential'. Defaults to 'linear'.
        coordinates_type: Determines are coordinates on a plane ('euclidean') or a sphere ('geographic').
            Used only in ordinary kriging. Defaults to 'geographic'.
        method: Ordinary or universal kriging. Defaults to 'ordinary'.

    Returns:
        Numpy array containing the interpolated values.

    Raises:
        EmptyDataFrameException: The input GeoDataFrame is empty.
        InvalidParameterValueException: Target column name is invalid or resolution is not greater than zero.
        NonMatchingCrsException: The input GeoDataFrame and raster profile have mismatching CRS.
    """

    if geodataframe.empty:
        raise EmptyDataFrameException("Expected GeoDataFrame to not be empty.")
    if raster_profile.get("crs") != geodataframe.crs:
        raise NonMatchingCrsException("Expected coordinate systems to match between raster and GeoDataFrame.")
    if target_column not in geodataframe.columns:
        raise InvalidParameterValueException(
            f"Expected target_column ({target_column}) to be contained in GeoDataFrame columns."
        )

    check_raster_profile(raster_profile)

    raster_width = raster_profile.get("width")
    raster_height = raster_profile.get("height")
    raster_transform = raster_profile.get("transform")

    data_interpolated = _kriging(
        geodataframe,
        target_column,
        raster_width,
        raster_height,
        raster_transform,
        variogram_model,
        coordinates_type,
        method,
    )

    return data_interpolated
