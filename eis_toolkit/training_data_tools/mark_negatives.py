import geopandas
import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Tuple, Union
from rasterio import profiles

from eis_toolkit.exceptions import EmptyDataFrameException, NonMatchingCrsException


def _mark_negatives(negatives, raster_array, raster_meta):

    out_array = np.copy(raster_array)

    n_row, n_cols = rasterio.transform.rowcol(raster_meta.get("transform"), negatives.geometry.x, negatives.geometry.y)

    out_array[np.ix_(n_row, n_cols)] = np.where(
        raster_array[np.ix_(n_row, n_cols)] == 0, -1, raster_array[np.ix_(n_row, n_cols)]
    )

    return out_array


@beartype
def mark_negatives(
    negatives: geopandas.GeoDataFrame,
    raster_array: np.ndarray,
    raster_meta: Union[profiles.Profile, dict],
) -> Tuple[np.ndarray, dict]:
    """Convert a point data set into a binary raster.

    Args:
        negatives: The geodataframe points set to be marked into raster.
        raster_array: The raster data with positives already marked.
        raster_meta: The raster metadata with crs.

    Returns:
        A tuple containing the output raster as a NumPy array and updated metadata.

    Raises:
        EmptyDataFrameException:  The input GeoDataFrame is empty.
        InvalidParameterValueException: Provided invalid input parameter.
        NonMatchingCrsException: The raster and geodataframe are not in the same CRS.
    """

    if negatives.empty:
        raise EmptyDataFrameException("Expected geodataframe to contain geometries.")

    if raster_meta.get("crs").to_epsg() != negatives.crs.to_epsg():
        raise NonMatchingCrsException("The metadata provided and geodataframe are not in the same CRS.")

    out_array = _mark_negatives(negatives, raster_array, raster_meta)

    return out_array, raster_meta
