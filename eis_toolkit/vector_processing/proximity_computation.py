from numbers import Number

import geopandas as gpd
import numpy as np
from beartype import beartype
from beartype.typing import Literal, Tuple, Union
from rasterio import profiles

from eis_toolkit.transformations.linear import _min_max_scaling
from eis_toolkit.vector_processing.distance_computation import distance_computation


@beartype
def proximity_computation(
    geodataframe: gpd.GeoDataFrame,
    raster_profile: Union[profiles.Profile, dict],
    maximum_distance: Number,
    scaling_method: Literal["linear"] = "linear",
    scale_range: Tuple[Number, Number] = (1, 0),
) -> np.ndarray:
    """Compute proximity to the nearest geometries.

    Args:
         geodataframe: The GeoDataFrame with geometries to determine proximity to.
         raster_profile: The raster profile of the raster in which the distances
                         to the nearest geometry are determined.
         max_distance: The maximum distance in the output array beyond which proximity is considered 0.
         scaling_method: Scaling method used to produce the proximity values. Defaults to 'linear'
         scaling_range: Min and max values used for scaling the proximity values. Defaults to (1,0).

    Returns:
         A 2D numpy array with the scaled values.

    Raises:
         NonMatchingCrsException: The input raster profile and geodataframe have mismatching CRS.
         EmptyDataFrameException: The input geodataframe is empty.

    """
    out_matrix = _linear_proximity_computation(geodataframe, raster_profile, maximum_distance, scale_range)

    return out_matrix


@beartype
def _linear_proximity_computation(
    geodataframe: gpd.GeoDataFrame,
    raster_profile: Union[profiles.Profile, dict],
    maximum_distance: Number,
    scaling_range: Tuple[Number, Number],
) -> np.ndarray:
    """Compute proximity to the nearest geometries.

    Args:
         geodataframe: The GeoDataFrame with geometries to determine proximity to.
         raster_profile: The raster profile of the raster in which the distances
                         to the nearest geometry are determined.
         max_distance: The maximum distance in the output array.
         scaling_range: a tuple of maximum value in the scaling and minimum value.

    Returns:
         A 2D numpy array with the linearly scaled values.

    Raises:
         NonMatchingCrsException: The input raster profile and geodataframe have mismatching CRS.
         EmptyDataFrameException: The input geodataframe is empty.
    """

    out_matrix = distance_computation(geodataframe, raster_profile, maximum_distance)

    out_matrix = _min_max_scaling(out_matrix, scaling_range)

    return out_matrix
