from numbers import Number

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.transform
from beartype import beartype
from beartype.typing import Tuple, Union
from rasterio import profiles
from shapely.geometry import Point

from eis_toolkit.exceptions import EmptyDataException


def _random_sampling(
    indices: np.ndarray,
    values: np.ndarray,
    sample_number: Number,
    random_seed: int,
) -> np.ndarray:

    indices_negatives = indices[values == 0]

    total_negatives = min(indices_negatives.size, sample_number)

    np.random.seed(random_seed)
    negative_indices = np.random.choice(indices_negatives.shape[0], total_negatives, replace=False)
    Negative_sample = indices_negatives[negative_indices]

    return Negative_sample


@beartype
def generate_negatives(
    raster_array: np.ndarray,
    raster_profile: Union[profiles.Profile, dict],
    sample_number: Number,
    random_seed: int = 48,
) -> Tuple[gpd.GeoDataFrame, np.ndarray, Union[profiles.Profile, dict]]:
    """Generate probable negatives from raster array with marked positives.

    Args:
        raster_array: Raster array with marked positives.
        raster_profile: The raster profile determining the output raster grid properties.
        sample_number: Maximum number of negatives to be generated.
        random_seed: Seed for generating random negatives.

    Returns:
        A tuple containing the shapely points, output raster as a NumPy array and updated metadata.

    Raises:
        EmptyDataException: The raster array is empty.
    """

    if raster_array.size == 0:
        raise EmptyDataException

    out_array = np.copy(raster_array)

    total_rows = out_array.shape[0]
    total_cols = out_array.shape[1]

    indices = np.arange(total_rows * total_cols)

    indices = indices.reshape(-1, 1)

    values = out_array.reshape(-1, 1)

    sampled_negatives = _random_sampling(
        indices=indices, values=values, sample_number=sample_number, random_seed=random_seed
    )

    sampled_negatives = sampled_negatives.reshape(1, -1)

    row = sampled_negatives // total_cols
    row = row[0]

    col = np.mod(sampled_negatives, total_cols)
    col = col[0]

    out_array[row, col] = -1

    x, y = rasterio.transform.xy(raster_profile["transform"], row, col)

    points = [Point(x[i], y[i]) for i in range(len(x))]

    sample_negative = gpd.GeoDataFrame(geometry=points)
    sample_negative.set_crs(raster_profile["crs"], allow_override=True, inplace=True)

    return sample_negative, out_array, raster_profile
