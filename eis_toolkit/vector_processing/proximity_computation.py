from numbers import Number

import geopandas as gpd
import numpy as np
from beartype import beartype
from beartype.typing import Tuple, Union
from rasterio import profiles

from eis_toolkit.transformations.linear import _min_max_scaling
from eis_toolkit.vector_processing.distance_computation import distance_computation


@beartype
def proximity_computation(
    geodataframe: gpd.GeoDataFrame,
    raster_profile: Union[profiles.Profile, dict],
    maximum_distance: Number,
    scale_range: Tuple[Number, Number] = (1, 0),
) -> Tuple[np.ndarray, Union[profiles.Profile, dict]]:
    """Compute proximity to the nearest geometries.

    Scales proximity values linearly in the given range. The first number in scale_range
    denotes the value at geometries, the second at given maximum_distance.

    Args:
        geodataframe: The GeoDataFrame with geometries to determine proximity to.
        raster_profile: The raster profile of the raster in which the distances to the
            nearest geometry are determined.
        max_distance: The maximum distance in the output array beyond which proximity is considered 0.
        scaling_range: Min and max values used for scaling the proximity values. Defaults to (1,0).

    Returns:
        A 2D numpy array with the linearly scaled proximity values and raster profile..

    Raises:
        NonMatchingCrsException: The input raster profile and geodataframe have mismatching CRS.
        EmptyDataFrameException: The input geodataframe is empty.
    """
    out_image, out_profile = _linear_proximity_computation(geodataframe, raster_profile, maximum_distance, scale_range)

    return out_image, out_profile


@beartype
def _linear_proximity_computation(
    geodataframe: gpd.GeoDataFrame,
    raster_profile: Union[profiles.Profile, dict],
    maximum_distance: Number,
    scaling_range: Tuple[Number, Number],
) -> Tuple[np.ndarray, Union[profiles.Profile, dict]]:
    out_image, out_profile = distance_computation(geodataframe, raster_profile, maximum_distance)

    out_image = _min_max_scaling(out_image, scaling_range)

    return out_image, out_profile
