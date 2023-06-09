from typing import Optional

import geopandas as gdp
import numpy as np
from beartype import beartype
from sklearn.cluster import KMeans

from eis_toolkit.exceptions import EmptyDataFrameException, InvalidParameterValueException


def _k_means_clustering(
    data: gdp.GeoDataFrame, number_of_clusters: Optional[int], random_state: Optional[int]
) -> gdp.GeoDataFrame:

    coordinates = list(data.geometry.apply(lambda geom: [geom.x, geom.y]))

    if number_of_clusters is None:
        # The elbow method
        k_max = 10
        inertia = np.array(
            [KMeans(n_clusters=k, random_state=0).fit(coordinates).inertia_ for k in range(1, k_max + 1)]
        )

        inertia = np.diff(inertia, 2)
        scaled_derivatives = [i * 100 for i in inertia]
        k_optimal = scaled_derivatives.index(min(scaled_derivatives))

        kmeans = KMeans(n_clusters=k_optimal, random_state=random_state)

    else:
        kmeans = KMeans(n_clusters=number_of_clusters, random_state=random_state)

    kmeans.fit(coordinates)
    data["cluster"] = kmeans.labels_

    return data


@beartype
def k_means_clustering(
    data: gdp.GeoDataFrame, number_of_clusters: Optional[int] = None, random_state: Optional[int] = None
) -> gdp.GeoDataFrame:
    """
    Perform k-means clustering on the input data.

    Args:
        data: A GeoDataFrame containing the input data.
        number_of_clusters: The number of clusters (>= 1) to form. Optional parameter. If not provided,
            optimal number of clusters is computed using the elbow method.
        random_state: A random number generation for centroid initialization to make
            the randomness deterministic. Optional parameter.

    Returns:
        GeoDataFrame containing assigned cluster labels.

    Raises:
        EmptyDataFrameException: The input GeoDataFrame is empty.
        InvalidParameterException: The number of clusters is less than one.
    """

    if data.empty:
        raise EmptyDataFrameException("The input GeoDataFrame is empty.")

    if number_of_clusters is not None:
        if number_of_clusters < 1:
            raise InvalidParameterValueException("The input value for number of clusters must be at least one.")

    k_means_gdf = _k_means_clustering(data, number_of_clusters, random_state)

    return k_means_gdf
