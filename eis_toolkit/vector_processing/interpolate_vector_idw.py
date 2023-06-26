from beartype import beartype
from beartype.typing import Optional, Union, Tuple
from scipy.spatial.distance import cdist
import numpy as np
import geopandas as gpd
from rasterio import profiles, transform

from eis_toolkit import exceptions


@beartype
def interpolate_vector(
    geodataframe: gpd.GeoDataFrame,
    resolution: Union[float, Tuple[float, float]],
    target_column: str,
    power: Optional[int] = 2,
    base_raster_profile: Optional[Union[profiles.Profile, dict]] = None,
) -> np.ndarray:
    """
    Interpolates data from a GeoDataFrame using Inverse Distance Weighting (IDW)
        based on the given target positions.

    Args:
        geodataframe: The input vector data.
        resolution: The cell size of the output raster.
        target_column: The target column for interpolation.
        power: The optional power parameter for IDW. Defaults to 2.
        base_raster_profile: Base raster profile
            to be used for determining the grid on which vectors are burned in.

    Returns:
        interpolated_values: A dictionary containing the interpolated values.
    """

    if geodataframe.shape[0] == 0:
        # Empty GeoDataFrame
        raise exceptions.EmptyDataFrameException("Expected geodataframe to contain geometries.")

    if target_column is None or target_column not in geodataframe.columns:
        raise exceptions.InvalidParameterValueException(
            f"Expected target_column ({target_column}) to be contained in geodataframe columns.")

    interpolated_values = {}
    values = geodataframe[target_column].values.astype(float)

    target_positions, positions, width, height = _create_target_positions(geodataframe, resolution, base_raster_profile)
    distances = cdist(positions, target_positions)
    distances[distances == 0] = np.finfo(float).eps  # Replace zeros with a small non-zero value
    weights = 1 / distances ** power
    weights /= weights.sum(axis=0)

    interpolated_values = np.dot(weights.T, values)
    interpolated_values.reshape(-1, 2)

    return interpolated_values


def _create_target_positions(geodataframe, resolution, base_raster_profile):
    if isinstance(resolution, tuple):
        pixel_width, pixel_height = resolution
    else:
        pixel_width = pixel_height = resolution

    if base_raster_profile is None:
        min_x, min_y, max_x, max_y = geodataframe.total_bounds
        width = int((max_x - min_x) / pixel_width)
        height = int((max_y - min_y) / pixel_height)
        out_transform = transform.from_bounds(min_x, min_y, max_x, max_y, width=width, height=height)
        target_positions = np.array(
            np.meshgrid(np.linspace(min_x, max_x, width), np.linspace(min_y, max_y, height))
        ).T.reshape(-1, 2)
    else:
        width, height, out_transform = (
            base_raster_profile["width"],
            base_raster_profile["height"],
            base_raster_profile["transform"],
        )
        min_x = out_transform[2]
        max_x = min_x + (width * out_transform[0])
        min_y = out_transform[5] + (height * out_transform[4])
        max_y = out_transform[5]
        target_positions = np.array(
            np.meshgrid(np.linspace(min_x, max_x, width), np.linspace(min_y, max_y, height))
        ).T.reshape(-1, 2)

    positions = np.array(geodataframe.geometry.apply(lambda geom: (geom.x, geom.y)).tolist())

    positions = positions.reshape(-1, 2)
    target_positions = target_positions.reshape(-1, 2)

    return target_positions, positions, width, height
