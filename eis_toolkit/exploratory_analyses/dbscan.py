import geopandas as gdp
import numpy as np
from beartype import beartype
from sklearn.cluster import DBSCAN

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidParameterValueException


def _dbscan(data: gdp.GeoDataFrame, max_distance: float, min_samples: int) -> gdp.GeoDataFrame:

    coordinates = list(data.geometry.apply(lambda geom: [geom.x, geom.y]))
    dbscan = DBSCAN(eps=max_distance, min_samples=min_samples)

    dbscan.fit(coordinates)
    data["cluster"] = dbscan.labels_

    core_indices = dbscan.core_sample_indices_
    core_data = np.zeros(len(coordinates), dtype=int)
    core_data[core_indices] = 1
    data["core"] = core_data

    return data


@beartype
def dbscan(data: gdp.GeoDataFrame, max_distance: float = 0.5, min_samples: int = 5) -> gdp.GeoDataFrame:
    """
    Perform DBSCAN clustering on the input data.

    Args:
        data: GeoDataFrame containing the input data.
        max_distance: The maximum distance between two samples for one to be considered as in the neighborhood of
            the other. Optional parameter.
        min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
            Optional parameter.

    Returns:
        GeoDataFrame containing assigned cluster labels and value 1 if a data point is a core point
        and 0 otherwise.

    Raises:
        EmptyDataFrameException: The input GeoDataFrame is empty.
        InvalidParameterException: The maximum distance between two samples in a neighborhood is not greater
            than zero or the number of samples in a neighborhood is not greater than one.
    """

    if data.empty:
        raise EmptyDataFrameException("The input GeoDataFrame is empty.")

    if max_distance <= 0:
        raise InvalidParameterValueException(
            "The input value for the maximum distance between two samples in a neighborhood must be greater than zero."
        )

    if min_samples <= 1:
        raise InvalidParameterValueException(
            "The input value for the minimum number of samples in a neighborhood must be greater than one."
        )

    dbscan_gdf = _dbscan(data, max_distance, min_samples)

    return dbscan_gdf
