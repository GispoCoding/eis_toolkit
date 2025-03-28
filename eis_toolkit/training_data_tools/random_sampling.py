import geopandas as gpd
import numpy as np
import rasterio
import rasterio.transform
from beartype import beartype
from beartype.typing import Tuple, Union
from rasterio import profiles
from shapely.geometry import Point

from eis_toolkit.exceptions import EmptyDataException, NumericValueSignException


def _random_sampling(
    indices: np.ndarray,
    values: np.ndarray,
    sample_number: int,
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
    raster_meta: Union[profiles.Profile, dict],
    sample_number: int,
    random_seed: int = 48,
) -> Tuple[gpd.GeoDataFrame, np.ndarray, Union[profiles.Profile, dict]]:
    """Generate probable negatives from binary raster array with marked positives.

    Generates a list of random negative points from a binary raster array,
    ensuring that these negatives do not overlap with the already marked positive 
    points. The positives can include points with or without attribute and radius, 
    as in the points_to_raster tool. 

    Args:
        raster_array: Binary raster array with marked positives.
        raster_meta: Raster metadata.
        sample_number: maximum number of negatives to be generated.
        random_seed: Seed for generating random negatives.

    Returns:
        A tuple containing the shapely points, output raster as a NumPy array and updated metadata.

    Raises:
        EmptyDataException: The raster array is empty.
        NumericValueSignException: The sample number is negative or zero.
    """

    if raster_array.size == 0:
        raise EmptyDataException("Expected non empty raster array.")
    
    if sample_number <= 0:
        raise NumericValueSignException("The sample number should be always be greater than zero")

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

    x, y = rasterio.transform.xy(raster_meta["transform"], row, col)

    points = [Point(x[i], y[i]) for i in range(len(x))]

    sample_negative = gpd.GeoDataFrame(geometry=points)
    sample_negative.set_crs(raster_meta["crs"], allow_override=True, inplace=True)

    return sample_negative, out_array, raster_meta
