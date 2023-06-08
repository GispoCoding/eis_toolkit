from typing import Optional

import geopandas as gdp
from beartype import beartype
from sklearn.cluster import KMeans

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidParameterValueException


def _k_means_clustering(
    data: gdp.GeoDataFrame, number_of_clusters: int, random_state: Optional[int]
) -> gdp.GeoDataFrame:

    coordinates = list(data.geometry.apply(lambda geom: [geom.x, geom.y]))

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=random_state)
    kmeans.fit(coordinates)

    data["cluster"] = kmeans.labels_

    return data


@beartype
def k_means_clustering(
    data: gdp.GeoDataFrame, number_of_clusters: int, random_state: Optional[int] = None
) -> gdp.GeoDataFrame:
    """
    Perform k-means clustering on the input data.

    Args:
        gdf: A GeoDataFrame containing the input data.
        number_of_clusters: The number of clusters to form.
        random_state: A random number generation for centroid initialization to make
            the randomness deterministic, default=None.

    Returns:
        GeoDataFrame containing assigned cluster labels.

    Raises:
        EmptyDataFrameException: The input GeoDataFrame is empty.
    """

    if data.empty:
        raise EmptyDataFrameException("The input GeoDataFrame is empty.")

    if number_of_clusters < 1:
        raise InvalidParameterValueException("The input value for number of clusters must be at least one.")

    k_means_gdf = _k_means_clustering(data, number_of_clusters, random_state)
    return k_means_gdf
